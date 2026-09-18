use super::*;

#[test]
fn previews_match_global_prefix_across_strides_permutations_and_source_types() {
    let maps = [
        vec![
            ComponentCoordinateMap::range(20, 0..8).unwrap(),
            ComponentCoordinateMap::range(20, 8..20).unwrap(),
        ],
        vec![ComponentCoordinateMap::indices(20, vec![19, 2, 16, 7, 0, 4, 1, 13, 10]).unwrap()],
        vec![
            ComponentCoordinateMap::indices(20, vec![1, 7, 13, 19]).unwrap(),
            ComponentCoordinateMap::indices(20, vec![4, 10, 16]).unwrap(),
        ],
    ];
    for data in [
        TensorObservationData::I64((0..60).map(|i| i64::MIN + i).collect()),
        TensorObservationData::U64((0..60).map(|i| u64::MAX - i).collect()),
        TensorObservationData::Bool((0..60).map(|i| i % 3 == 0).collect()),
        TensorObservationData::F32(
            (0..60)
                .map(|i| [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.5][i % 5])
                .collect(),
        ),
    ] {
        for max_elements in [1, 6, 7, 8, 14, 30] {
            let plan = plan_for(CaptureTransform::Preview { max_elements }, false);
            let global = Value {
                shape: vec![3, 20],
                data: data.clone(),
            };
            let mut ordinary = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(plan.clone()));
            ordinary.begin_step(CapturePhase::Prefill, 0).unwrap();
            ordinary
                .observe(&mut Backend::default(), "block.output", &global)
                .unwrap();
            let expected = ordinary.take_step().unwrap().records.remove(0);
            for maps in &maps {
                let mut ledger = CaptureLedger::new(&plan);
                let receipt = authority(&plan, maps, &mut ledger);
                let encoded = encode_all(&plan, &receipt, &global, maps, &mut ledger);
                let mut delivery = receipt.into_delivery();
                for (rank, bytes) in encoded.iter().enumerate().rev() {
                    delivery.receive(rank, bytes, &mut ledger).unwrap();
                }
                let capture = delivery.finish(&mut ledger).unwrap();
                let actual = capture.capture().record();
                assert_eq!(actual.outcome, expected.outcome);
                assert_eq!(actual.selected_shape, Some(vec![2, 7]));
                assert_eq!(actual.source_dtype, expected.source_dtype);
                assert_eq!(
                    serde_json::to_value(&actual.payload).unwrap(),
                    serde_json::to_value(&expected.payload).unwrap()
                );
                let Some(CapturePayload::Tensor(value)) = &actual.payload else {
                    panic!()
                };
                assert_eq!(value.shape(), [max_elements.min(14) as usize]);
            }
        }
    }
}

#[test]
fn preview_receipts_reject_incorrect_truncation_and_payload_geometry() {
    let plan = plan_for(CaptureTransform::Preview { max_elements: 3 }, false);
    let maps = [
        ComponentCoordinateMap::range(20, 0..8).unwrap(),
        ComponentCoordinateMap::range(20, 8..20).unwrap(),
    ];
    let global = Value {
        shape: vec![3, 20],
        data: TensorObservationData::U64((0..60).collect()),
    };
    let receipt = authority(&plan, &maps, &mut CaptureLedger::new(&plan));
    let encoded = encode_all(
        &plan,
        &receipt,
        &global,
        &maps,
        &mut CaptureLedger::new(&plan),
    );
    let original: PartitionCaptureProducerRecord = serde_json::from_slice(&encoded[0]).unwrap();
    for mutation in 0..5 {
        let mut changed = original.clone();
        let record = &mut changed.fragments[0].record;
        match mutation {
            0 => record.outcome = CaptureOutcome::Captured,
            1 => {
                record.outcome = CaptureOutcome::Truncated {
                    available_elements: 7,
                    emitted_elements: 3,
                }
            }
            2 => {
                record.outcome = CaptureOutcome::Truncated {
                    available_elements: 6,
                    emitted_elements: 2,
                }
            }
            3 => {
                record.payload = Some(CapturePayload::Tensor(
                    TensorObservation::new(vec![1, 3], TensorObservationData::U64(vec![1, 4, 7]))
                        .unwrap(),
                ))
            }
            _ => {
                record.payload = Some(CapturePayload::Tensor(
                    TensorObservation::new(vec![2], TensorObservationData::U64(vec![1, 4]))
                        .unwrap(),
                ))
            }
        }
        let mut ledger = CaptureLedger::new(&plan);
        let mut delivery = authority(&plan, &maps, &mut ledger).into_delivery();
        assert!(delivery
            .receive(0, &serde_json::to_vec(&changed).unwrap(), &mut ledger)
            .is_err());
        assert_eq!(delivery.missing_producers().collect::<Vec<_>>(), [0, 1]);
    }
}

#[test]
fn preview_assembly_budget_cannot_refund_completed_fragment_exports() {
    let maps = [
        ComponentCoordinateMap::range(20, 0..8).unwrap(),
        ComponentCoordinateMap::range(20, 8..20).unwrap(),
    ];
    let global = Value {
        shape: vec![3, 20],
        data: TensorObservationData::I64((0..60).collect()),
    };
    let transform = CaptureTransform::Preview { max_elements: 3 };
    let generous = transformed_plan(1_000_000, transform.clone());
    let mut measured = CaptureLedger::new(&generous);
    capture_maps(
        &generous,
        &global,
        &maps,
        &mut Backend::default(),
        &mut measured,
    );
    let limited = transformed_plan(measured.total().host_bytes, transform);
    let mut ledger = CaptureLedger::new(&limited);
    let mut backend = Backend::default();
    let fragments = capture_maps(&limited, &global, &maps, &mut backend, &mut ledger);
    let consumed = ledger.total();
    assert_eq!(backend.exported, 6);
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
    assert_eq!(ledger.total(), consumed);
}

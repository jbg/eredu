mod exchange;
mod preview;
mod routed;
mod sum;
mod vocabulary;
use super::*;

fn context(plan: &AdmittedCapturePlan) -> PartitionCaptureContext {
    PartitionCaptureContext {
        invocation: None,
        artifact_identity: "artifact-exact".into(),
        execution_identity: "execution-version-3".into(),
        run_identity: "branch-7".into(),
        overlay_identity: Some("overlay-coordinated-2".into()),
        capture_plan_identity: plan.identity().into(),
        selection_index: 0,
        phase: CapturePhase::Prefill,
        prediction: 0,
        forward_epoch: 19,
    }
}

fn plan_for(transform: CaptureTransform, empty: bool) -> AdmittedCapturePlan {
    let (mut plan, mut catalog, support, capabilities) = fixture(transform);
    catalog.points[0].axes.as_mut().unwrap()[1].dimension = SymbolicDimension::Known(20);
    plan.selections[0].schedule.decode = false;
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
            end: if empty { 1 } else { 20 },
            stride: 3,
        },
    ];
    plan.limits.per_step.host_bytes = 32_000_000;
    plan.limits.cumulative.host_bytes = 32_000_000;
    admit(plan, &catalog, &support, &capabilities).unwrap()
}

fn authority(
    plan: &AdmittedCapturePlan,
    maps: &[ComponentCoordinateMap],
    ledger: &mut CaptureLedger,
) -> PartitionCaptureReceiptPlan {
    let slice = resolve_slice(&plan.points()[0], &plan.plan().selections[0], &[3, 20]).unwrap();
    PartitionCaptureReceiptPlan::new(
        plan.clone(),
        context(plan),
        maps.iter()
            .enumerate()
            .map(|(rank, map)| PartitionCaptureProducer {
                rank,
                projection: CaptureSlicePartition::new(&[3, 20], &slice, 1, map, 16).unwrap(),
            })
            .collect(),
        maps.len(),
        PartitionCaptureReceiptLimits {
            max_producers: 8,
            max_fragments: 16,
            max_record_bytes: 16_384,
        },
        ledger,
    )
    .unwrap()
}

fn encode_all(
    plan: &AdmittedCapturePlan,
    authority: &PartitionCaptureReceiptPlan,
    global: &Value,
    maps: &[ComponentCoordinateMap],
    ledger: &mut CaptureLedger,
) -> Vec<Vec<u8>> {
    let fragments = capture_maps(plan, global, maps, &mut Backend::default(), ledger);
    maps.iter()
        .enumerate()
        .map(|(rank, _)| {
            authority
                .encode_producer(
                    rank,
                    Backend::default().source_dtype(global),
                    fragments
                        .iter()
                        .filter(|fragment| fragment.producer_rank() == rank)
                        .cloned()
                        .collect(),
                    ledger,
                )
                .unwrap()
        })
        .collect()
}

fn maps_with_idle_producer() -> Vec<ComponentCoordinateMap> {
    vec![
        ComponentCoordinateMap::range(20, 0..8).unwrap(),
        ComponentCoordinateMap::range(20, 8..10).unwrap(), // no selected scalar in this interval
        ComponentCoordinateMap::range(20, 10..20).unwrap(),
    ]
}

#[test]
fn producer_receipts_roundtrip_exact_values_and_retain_execution_context() {
    for data in [
        TensorObservationData::I64((0..60).map(|i| i64::MIN + i).collect()),
        TensorObservationData::U64((0..60).map(|i| u64::MAX - i).collect()),
        TensorObservationData::Bool((0..60).map(|i| i % 3 == 0).collect()),
        TensorObservationData::F32((0..60).map(|i| i as f32 * 0.25 - 3.0).collect()),
    ] {
        let plan = plan_for(CaptureTransform::Slice, false);
        let global = Value {
            shape: vec![3, 20],
            data,
        };
        let maps = maps_with_idle_producer();
        let mut ledger = CaptureLedger::new(&plan);
        let authority = authority(&plan, &maps, &mut ledger);
        let records = encode_all(&plan, &authority, &global, &maps, &mut ledger);
        let mut delivery = authority.into_delivery();
        assert_eq!(delivery.missing_producers().collect::<Vec<_>>(), [0, 1, 2]);
        for rank in [2, 0, 1] {
            delivery.receive(rank, &records[rank], &mut ledger).unwrap();
        }
        assert!(delivery.missing_producers().next().is_none());
        let result = delivery.finish(&mut ledger).unwrap();
        assert_eq!(result.context(), &context(&plan));
        assert_eq!(result.producers(), [0, 1, 2]);
        assert_eq!(result.capture().contributions().len(), 2);
        let mut ordinary = CaptureSession::new(plan.clone());
        ordinary.begin_step(CapturePhase::Prefill, 0).unwrap();
        ordinary
            .observe(&mut Backend::default(), "block.output", &global)
            .unwrap();
        let expected = ordinary.take_step().unwrap();
        assert_eq!(
            result.capture().record().payload,
            expected.records[0].payload
        );
        assert!(
            ledger.total().host_bytes > result.capture().record().charged.host_bytes,
            "wire/decode storage is additionally reserved"
        );
    }
}

#[test]
fn producer_receipts_preserve_nonfinite_wire_values_and_reduced_results() {
    for transform in [
        CaptureTransform::Slice,
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![-2.0, 0.0, 2.0],
        },
    ] {
        let plan = plan_for(transform.clone(), false);
        let mut values = (0..60)
            .map(|i| (i as f32 - 30.0) * 0.25)
            .collect::<Vec<_>>();
        values[1] = f32::NAN;
        values[4] = f32::INFINITY;
        values[7] = f32::NEG_INFINITY;
        let global = Value {
            shape: vec![3, 20],
            data: TensorObservationData::F32(values.clone()),
        };
        let maps =
            [ComponentCoordinateMap::indices(20, vec![19, 2, 16, 7, 0, 4, 1, 13, 10]).unwrap()];
        let mut ledger = CaptureLedger::new(&plan);
        let authority = authority(&plan, &maps, &mut ledger);
        let records = encode_all(&plan, &authority, &global, &maps, &mut ledger);
        let mut delivery = authority.into_delivery();
        delivery.receive(0, &records[0], &mut ledger).unwrap();
        let result = delivery.finish(&mut ledger).unwrap();
        let slice =
            resolve_slice(&plan.points()[0], &plan.plan().selections[0], &global.shape).unwrap();
        let expected = selected_indices(&global.shape, &slice)
            .into_iter()
            .map(|i| values[i])
            .collect::<Vec<_>>();
        let payload = result.capture().record().payload.as_ref().unwrap();
        if let CapturePayload::Tensor(value) = payload {
            let TensorObservationData::F32(actual) = value.data() else {
                panic!()
            };
            for (a, b) in actual.iter().zip(expected) {
                assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()));
            }
        } else {
            assert_reduced_close(payload, &reference_reduction(&expected, &transform));
        }
    }
}

#[test]
fn receipt_admission_rejects_ownership_gaps_and_overlap_before_capture() {
    let plan = plan_for(CaptureTransform::Slice, false);
    let slice = resolve_slice(&plan.points()[0], &plan.plan().selections[0], &[3, 20]).unwrap();
    for ranges in [vec![0..8], vec![0..11, 8..20]] {
        let mut ledger = CaptureLedger::new(&plan);
        let result = PartitionCaptureReceiptPlan::new(
            plan.clone(),
            context(&plan),
            ranges
                .into_iter()
                .enumerate()
                .map(|(rank, range)| PartitionCaptureProducer {
                    rank,
                    projection: CaptureSlicePartition::new(
                        &[3, 20],
                        &slice,
                        1,
                        &ComponentCoordinateMap::range(20, range).unwrap(),
                        16,
                    )
                    .unwrap(),
                })
                .collect(),
            2,
            PartitionCaptureReceiptLimits {
                max_producers: 2,
                max_fragments: 4,
                max_record_bytes: 16_384,
            },
            &mut ledger,
        );
        assert!(matches!(
            result,
            Err(PartitionCaptureMergeError::Incomplete { .. })
                | Err(PartitionCaptureMergeError::Overlap)
        ));
        assert_eq!(ledger.total(), CaptureUsage::default());
    }
}

#[test]
fn receipts_reject_wrong_senders_stale_context_duplicate_ordinals_and_replay() {
    let plan = plan_for(CaptureTransform::Slice, false);
    let global = Value {
        shape: vec![3, 20],
        data: TensorObservationData::F32(vec![1.0; 60]),
    };
    let maps = [ComponentCoordinateMap::indices(20, vec![19, 2, 16, 7, 0, 4, 1, 13, 10]).unwrap()];
    let authority = authority(&plan, &maps, &mut CaptureLedger::new(&plan));
    let bytes = encode_all(
        &plan,
        &authority,
        &global,
        &maps,
        &mut CaptureLedger::new(&plan),
    )
    .remove(0);
    let original: PartitionCaptureProducerRecord = serde_json::from_slice(&bytes).unwrap();
    for mutation in 0..19 {
        let mut ledger = CaptureLedger::new(&plan);
        let mut delivery = super::receipts::authority(&plan, &maps, &mut ledger).into_delivery();
        let mut changed = original.clone();
        match mutation {
            0 => changed.schema_version += 1,
            1 => changed.context.artifact_identity.push('x'),
            2 => changed.context.execution_identity.push('x'),
            3 => changed.context.run_identity.push('x'),
            4 => changed.context.overlay_identity = None,
            5 => changed.context.capture_plan_identity.push('x'),
            6 => changed.context.forward_epoch += 1,
            7 => changed.context.prediction += 1,
            8 => changed.producer_rank = 1,
            9 => changed.fragments[1].fragment_index = changed.fragments[0].fragment_index,
            10 => {
                changed.fragments.pop();
            }
            11 => changed.fragments[0].record.source_shape = Some(vec![3, 20]),
            12 => changed.fragments[0].record.path.push('x'),
            13 => changed.receipt_plan_identity.push('x'),
            14 => {
                let Some(CapturePayload::Tensor(value)) = &changed.fragments[0].record.payload
                else {
                    panic!()
                };
                changed.fragments[0].record.payload = Some(CapturePayload::Tensor(
                    TensorObservation::new(
                        value.shape().to_vec(),
                        TensorObservationData::I64(vec![0; value.data().len()]),
                    )
                    .unwrap(),
                ));
            }
            15 => {
                changed.fragments[0].record.charged.host_bytes =
                    plan.plan().limits.per_step.host_bytes + 1
            }
            16 => changed.fragments[0].record.payload = None,
            17 => changed.context.selection_index += 1,
            18 => changed.context.phase = CapturePhase::Decode,
            _ => unreachable!(),
        }
        let changed = serde_json::to_vec(&changed).unwrap();
        let before = ledger.total();
        assert!(
            delivery.receive(0, &changed, &mut ledger).is_err(),
            "mutation {mutation}"
        );
        assert!(
            ledger.total().host_bytes > before.host_bytes,
            "failed decoding remains charged"
        );
        assert_eq!(delivery.missing_producers().collect::<Vec<_>>(), [0]);
    }
    let mut ledger = CaptureLedger::new(&plan);
    let mut delivery = authority.into_delivery();
    let before = ledger.total();
    assert!(delivery.receive(99, &bytes, &mut ledger).is_err());
    assert_eq!(
        ledger.total(),
        before,
        "wrong sender rejected before parser allocation"
    );
    delivery.receive(0, &bytes, &mut ledger).unwrap();
    let before = ledger.total();
    assert!(delivery.receive(0, &bytes, &mut ledger).is_err());
    assert_eq!(
        ledger.total(),
        before,
        "duplicate receipt rejected before parser allocation"
    );
}

#[test]
fn empty_selection_needs_explicit_receipts_and_preserves_dtype() {
    for (transform, dtype) in [
        (CaptureTransform::Slice, TensorDtype::U64),
        (CaptureTransform::Slice, TensorDtype::Bf16),
        (
            CaptureTransform::Preview { max_elements: 3 },
            TensorDtype::U64,
        ),
        (CaptureTransform::Summary, TensorDtype::F32),
        (
            CaptureTransform::Histogram {
                edges: vec![-1.0, 0.0, 1.0],
            },
            TensorDtype::F32,
        ),
    ] {
        let plan = plan_for(transform, true);
        let maps = [
            ComponentCoordinateMap::range(20, 0..10).unwrap(),
            ComponentCoordinateMap::range(20, 10..20).unwrap(),
        ];
        let mut ledger = CaptureLedger::new(&plan);
        let authority = authority(&plan, &maps, &mut ledger);
        let records = (0..2)
            .map(|rank| {
                authority
                    .encode_producer(rank, Some(dtype.clone()), Vec::new(), &mut ledger)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut partial = super::receipts::authority(&plan, &maps, &mut ledger).into_delivery();
        partial.receive(0, &records[0], &mut ledger).unwrap();
        assert!(matches!(
            partial.finish(&mut ledger),
            Err(PartitionCaptureMergeError::MissingProducer { producer_rank: 1 })
        ));
        let mut delivery = authority.into_delivery();
        for rank in 0..2 {
            delivery.receive(rank, &records[rank], &mut ledger).unwrap();
        }
        let result = delivery.finish(&mut ledger).unwrap();
        assert_eq!(result.producers(), [0, 1]);
        assert_eq!(result.capture().record().source_dtype, Some(dtype));
        assert_eq!(result.capture().record().selected_shape, Some(vec![2, 0]));
        match result.capture().record().payload.as_ref().unwrap() {
            CapturePayload::Tensor(value) => assert!(value.data().is_empty()),
            CapturePayload::Summary(value) => {
                assert_eq!(value.elements, 0);
                assert_eq!(value.mean, None);
            }
            CapturePayload::Histogram(value) => assert_eq!(value.counts, [0, 0]),
            _ => panic!(),
        }
    }
}

#[test]
fn decoding_limits_are_reserved_before_parsing_and_never_refunded() {
    let plan = plan_for(CaptureTransform::Slice, false);
    let maps = [ComponentCoordinateMap::range(20, 0..20).unwrap()];
    let authority = authority(&plan, &maps, &mut CaptureLedger::new(&plan));
    let mut delivery = authority.into_delivery();
    let limited = super::plan(1);
    let mut ledger = CaptureLedger::new(&limited);
    assert!(matches!(
        delivery.receive(0, b"{invalid", &mut ledger),
        Err(CaptureError::Limit {
            budget: CaptureBudget::Host,
            ..
        })
    ));
    assert_eq!(ledger.total(), CaptureUsage::default());
    let mut ledger = CaptureLedger::new(&plan);
    assert!(delivery
        .receive(0, &vec![b' '; 16_385], &mut ledger)
        .is_err());
    assert_eq!(ledger.total(), CaptureUsage::default());
    assert!(delivery.receive(0, b"{invalid", &mut ledger).is_err());
    assert!(ledger.total().host_bytes > 0);
}

#[test]
fn receipt_identity_rejects_same_shape_but_different_global_rank_ownership() {
    let (mut raw, mut catalog, support, capabilities) = fixture(CaptureTransform::FullTensor);
    catalog.points[0].axes.as_mut().unwrap()[1].dimension = SymbolicDimension::Known(20);
    raw.limits.per_step.host_bytes = 32_000_000;
    raw.limits.cumulative.host_bytes = 32_000_000;
    let plan = admit(raw, &catalog, &support, &capabilities).unwrap();
    let expected = [
        ComponentCoordinateMap::range(20, 0..10).unwrap(),
        ComponentCoordinateMap::range(20, 10..20).unwrap(),
    ];
    let changed = [expected[1].clone(), expected[0].clone()];
    let mut ledger = CaptureLedger::new(&plan);
    let sender = authority(&plan, &changed, &mut ledger);
    let receiver = authority(&plan, &expected, &mut ledger);
    assert_ne!(sender.identity(), receiver.identity());
    assert_eq!(
        sender.producer(0).unwrap().local_shape(),
        receiver.producer(0).unwrap().local_shape()
    );
    let global = Value {
        shape: vec![3, 20],
        data: TensorObservationData::F32((0..60).map(|i| i as f32 + 1.0).collect()),
    };
    let records = encode_all(&plan, &sender, &global, &changed, &mut ledger);
    let mut delivery = receiver.into_delivery();
    assert!(delivery.receive(0, &records[0], &mut ledger).is_err());
    assert_eq!(delivery.missing_producers().collect::<Vec<_>>(), [0, 1]);
}

#[test]
fn non_value_producer_outcomes_cannot_be_assembled_as_measured_zero() {
    let plan = plan_for(CaptureTransform::Slice, false);
    let maps = [ComponentCoordinateMap::range(20, 0..20).unwrap()];
    let global = Value {
        shape: vec![3, 20],
        data: TensorObservationData::F32(vec![2.0; 60]),
    };
    let sender = authority(&plan, &maps, &mut CaptureLedger::new(&plan));
    let bytes = encode_all(
        &plan,
        &sender,
        &global,
        &maps,
        &mut CaptureLedger::new(&plan),
    )
    .remove(0);
    for outcome in [
        CaptureOutcome::Missing,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::Limit {
                budget: CaptureBudget::Host,
                cumulative: true,
            },
        },
        CaptureOutcome::Failed {
            reason: CaptureFailureReason::Native,
            message: "native transform failed".into(),
        },
    ] {
        let mut receipt: PartitionCaptureProducerRecord = serde_json::from_slice(&bytes).unwrap();
        receipt.fragments[0].record.outcome = outcome.clone();
        receipt.fragments[0].record.payload = None;
        let mut ledger = CaptureLedger::new(&plan);
        let mut delivery = authority(&plan, &maps, &mut ledger).into_delivery();
        delivery
            .receive(0, &serde_json::to_vec(&receipt).unwrap(), &mut ledger)
            .unwrap();
        match delivery.finish(&mut ledger) {
            Err(PartitionCaptureMergeError::FragmentOutcome {
                producer_rank,
                outcome: actual,
            }) => {
                assert_eq!(producer_rank, 0);
                assert_eq!(actual, outcome);
            }
            other => panic!("non-value receipt became a value: {other:?}"),
        }
    }
}

#[test]
fn deferred_skipped_producer_retains_its_outcome_without_fabricating_source_precision() {
    let plan = plan_for(CaptureTransform::Slice, false);
    let maps = [
        ComponentCoordinateMap::range(20, 0..8).unwrap(),
        ComponentCoordinateMap::range(20, 8..20).unwrap(),
    ];
    let global = Value {
        shape: vec![3, 20],
        data: TensorObservationData::F32(vec![2.0; 60]),
    };
    let mut ledger = CaptureLedger::new(&plan);
    let authority = authority(&plan, &maps, &mut ledger);
    let mut bytes = encode_all(&plan, &authority, &global, &maps, &mut ledger);
    let mut skipped: PartitionCaptureProducerRecord = serde_json::from_slice(&bytes[1]).unwrap();
    let outcome = CaptureOutcome::Skipped {
        reason: CaptureSkipReason::Limit {
            budget: CaptureBudget::Retention,
            cumulative: true,
        },
    };
    skipped.source_dtype = None;
    for fragment in &mut skipped.fragments {
        fragment.record.source_dtype = None;
        fragment.record.payload = None;
        fragment.record.outcome = outcome.clone();
    }
    bytes[1] = serde_json::to_vec(&skipped).unwrap();
    let mut delivery = authority.into_delivery();
    for rank in 0..2 {
        delivery.receive(rank, &bytes[rank], &mut ledger).unwrap();
    }
    match delivery.finish(&mut ledger) {
        Err(PartitionCaptureMergeError::FragmentOutcome {
            producer_rank: 1,
            outcome: actual,
        }) => assert_eq!(actual, outcome),
        other => panic!("deferred skip lost its attribution: {other:?}"),
    }
}

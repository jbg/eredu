use super::*;

fn sum_authority(
    plan: &AdmittedCapturePlan,
    ranks: usize,
    ledger: &mut CaptureLedger,
) -> PartitionCaptureReceiptPlan {
    let slice = resolve_slice(&plan.points()[0], &plan.plan().selections[0], &[3, 20]).unwrap();
    PartitionCaptureReceiptPlan::new_sum(
        eredu_core::capture::SharedCapturePlan::new(plan.clone()),
        context(plan),
        (0..ranks)
            .map(|rank| PartitionCaptureProducer {
                rank,
                projection: CaptureSlicePartition::new(
                    &[3, 20],
                    &slice,
                    1,
                    &ComponentCoordinateMap::range(20, 0..20).unwrap(),
                    1,
                )
                .unwrap(),
            })
            .collect(),
        ranks,
        PartitionCaptureReceiptLimits {
            max_producers: ranks,
            max_fragments: ranks,
            max_record_bytes: 16_384,
        },
        ledger,
    )
    .unwrap()
}

fn terms(nonfinite: bool) -> (Vec<Value>, Vec<f32>) {
    let mut expected = (0..60)
        .map(|index| index as f32 * 0.25 - 3.0)
        .collect::<Vec<_>>();
    // F32 accumulation would erase every surviving nonzero middle term.
    let mut terms = vec![vec![1e20; 60], expected.clone(), vec![-1e20; 60]];
    if nonfinite {
        terms[0][1] = f32::INFINITY;
        terms[2][1] = f32::NEG_INFINITY;
        expected[1] = f32::NAN;
        terms[0][4] = f32::INFINITY;
        expected[4] = f32::INFINITY;
        terms[2][7] = f32::NEG_INFINITY;
        expected[7] = f32::NEG_INFINITY;
    }
    (
        terms
            .into_iter()
            .map(|values| Value {
                shape: vec![3, 20],
                data: TensorObservationData::F32(values),
            })
            .collect(),
        expected,
    )
}

fn encode_terms(
    plan: &AdmittedCapturePlan,
    authority: &PartitionCaptureReceiptPlan,
    terms: &[Value],
    backend: &mut Backend,
    ledger: &mut CaptureLedger,
) -> Vec<Vec<u8>> {
    terms
        .iter()
        .enumerate()
        .map(|(rank, value)| {
            let projection = authority.producer(rank).unwrap();
            let fragments = (0..projection.fragments().len())
                .map(|fragment_index| {
                    capture_sum_fragment(
                        backend,
                        value,
                        PartitionCaptureRequest {
                            invocation: None,
                            plan,
                            selection_index: 0,
                            phase: CapturePhase::Prefill,
                            prediction: 0,
                            projection,
                            fragment_index,
                            producer_rank: rank,
                        },
                        ledger,
                    )
                    .unwrap()
                })
                .collect();
            authority
                .encode_producer(rank, backend.source_dtype(value), fragments, ledger)
                .unwrap()
        })
        .collect()
}

#[test]
fn summed_receipts_preserve_cancellation_and_transform_only_the_complete_value() {
    for transform in [
        CaptureTransform::Slice,
        CaptureTransform::Preview { max_elements: 5 },
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![-2.0, 0.0, 2.0],
        },
    ] {
        for nonfinite in [false, true] {
            let plan = plan_for(transform.clone(), false);
            let mut ledger = CaptureLedger::new(&plan);
            let authority = sum_authority(&plan, 3, &mut ledger);
            assert_eq!(
                authority.combination(),
                PartitionCaptureCombination::SumF64ToF32
            );
            let (terms, expected) = terms(nonfinite);
            let mut backend = Backend::default();
            let records = encode_terms(&plan, &authority, &terms, &mut backend, &mut ledger);
            assert_eq!(backend.transforms, 3);
            assert_eq!(
                backend.exported,
                if matches!(transform, CaptureTransform::Preview { .. }) {
                    15
                } else {
                    42
                }
            );
            let mut delivery = authority.into_delivery();
            for rank in [2, 0, 1] {
                delivery.receive(rank, &records[rank], &mut ledger).unwrap();
            }
            let result = delivery.finish(&mut ledger).unwrap();
            assert_eq!(
                result.capture().combination(),
                PartitionCaptureCombination::SumF64ToF32
            );
            assert_eq!(result.producers(), [0, 1, 2]);
            assert_eq!(result.capture().contributions().len(), 3);
            let slice =
                resolve_slice(&plan.points()[0], &plan.plan().selections[0], &[3, 20]).unwrap();
            let selected = selected_indices(&[3, 20], &slice)
                .into_iter()
                .map(|index| expected[index])
                .collect::<Vec<_>>();
            let payload = result.capture().record().payload.as_ref().unwrap();
            if let CapturePayload::Tensor(tensor) = payload {
                let TensorObservationData::F32(actual) = tensor.data() else {
                    panic!()
                };
                let emitted = if matches!(transform, CaptureTransform::Preview { .. }) {
                    5
                } else {
                    selected.len()
                };
                assert_eq!(actual.len(), emitted);
                for (actual, expected) in actual.iter().zip(selected.iter().take(emitted)) {
                    assert!(
                        actual.to_bits() == expected.to_bits()
                            || actual.is_nan() && expected.is_nan(),
                        "{actual} vs {expected}"
                    );
                }
            } else {
                assert_reduced_close(payload, &reference_reduction(&selected, &transform));
            }
        }
    }
}

#[test]
fn sum_authority_rejects_partial_coverage_disjoint_terms_and_forged_wire_equations() {
    let plan = plan_for(CaptureTransform::Slice, false);
    let slice = resolve_slice(&plan.points()[0], &plan.plan().selections[0], &[3, 20]).unwrap();
    let mut ledger = CaptureLedger::new(&plan);
    let limits = PartitionCaptureReceiptLimits {
        max_producers: 3,
        max_fragments: 3,
        max_record_bytes: 16_384,
    };
    let producers = |ranges: Vec<std::ops::Range<usize>>| {
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
                    1,
                )
                .unwrap(),
            })
            .collect()
    };
    assert!(PartitionCaptureReceiptPlan::new_sum(
        eredu_core::capture::SharedCapturePlan::new(plan.clone()),
        context(&plan),
        producers(vec![0..10, 10..20]),
        2,
        limits,
        &mut ledger
    )
    .is_err());
    assert!(matches!(
        PartitionCaptureReceiptPlan::new(
            eredu_core::capture::SharedCapturePlan::new(plan.clone()),
            context(&plan),
            producers(vec![0..20, 0..20]),
            2,
            limits,
            &mut ledger
        ),
        Err(PartitionCaptureMergeError::Overlap)
    ));
    let authority = sum_authority(&plan, 3, &mut ledger);
    let (terms, _) = terms(false);
    let ordinary = capture_fragment(
        &mut Backend::default(),
        &terms[0],
        PartitionCaptureRequest {
            invocation: None,
            plan: &plan,
            selection_index: 0,
            phase: CapturePhase::Prefill,
            prediction: 0,
            projection: authority.producer(0).unwrap(),
            fragment_index: 0,
            producer_rank: 0,
        },
        &mut ledger,
    )
    .unwrap();
    assert!(authority
        .encode_producer(0, Some(TensorDtype::F32), vec![ordinary], &mut ledger)
        .is_err());
    let records = encode_terms(
        &plan,
        &authority,
        &terms,
        &mut Backend::default(),
        &mut ledger,
    );
    let mut forged: PartitionCaptureProducerRecord = serde_json::from_slice(&records[0]).unwrap();
    assert_eq!(forged.schema_version, 3);
    assert_eq!(forged.combination, PartitionCaptureCombination::SumF64ToF32);
    forged.combination = PartitionCaptureCombination::Disjoint;
    let mut delivery = authority.into_delivery();
    assert!(delivery
        .receive(0, &serde_json::to_vec(&forged).unwrap(), &mut ledger)
        .is_err());
    delivery.receive(0, &records[0], &mut ledger).unwrap();
    assert!(delivery.receive(0, &records[0], &mut ledger).is_err());
    delivery.receive(2, &records[2], &mut ledger).unwrap();
    assert!(matches!(
        delivery.finish(&mut ledger),
        Err(PartitionCaptureMergeError::MissingProducer { producer_rank: 1 })
    ));
}

#[test]
fn summed_receipts_require_empty_peer_acknowledgments_and_reserve_before_native_export() {
    let plan = plan_for(CaptureTransform::Summary, true);
    let mut ledger = CaptureLedger::new(&plan);
    let authority = sum_authority(&plan, 3, &mut ledger);
    let (terms, _) = terms(false);
    let mut backend = Backend::default();
    let records = encode_terms(&plan, &authority, &terms, &mut backend, &mut ledger);
    assert_eq!(backend.transforms, 0);
    let mut delivery = authority.into_delivery();
    for rank in [1, 2, 0] {
        delivery.receive(rank, &records[rank], &mut ledger).unwrap();
    }
    let result = delivery.finish(&mut ledger).unwrap();
    assert_eq!(
        result.capture().combination(),
        PartitionCaptureCombination::SumF64ToF32
    );
    assert_eq!(result.producers(), [0, 1, 2]);
    assert!(
        matches!(&result.capture().record().payload, Some(CapturePayload::Summary(summary)) if summary.elements == 0)
    );

    struct RejectNative {
        priced_values: u64,
    }
    impl CaptureReservation for RejectNative {
        fn reserve(
            &mut self,
            usage: CaptureUsage,
        ) -> Result<Option<CaptureSkipReason>, CaptureError> {
            if usage.retained_bytes != 0 {
                self.priced_values = usage.host_bytes / 8;
                return Ok(Some(CaptureSkipReason::Limit {
                    budget: CaptureBudget::Host,
                    cumulative: false,
                }));
            }
            Ok(None)
        }
    }
    let plan = plan_for(CaptureTransform::Summary, false);
    let mut ledger = CaptureLedger::new(&plan);
    let authority = sum_authority(&plan, 3, &mut ledger);
    let mut rejected = RejectNative { priced_values: 0 };
    let result = capture_sum_fragment(
        &mut backend,
        &terms[0],
        PartitionCaptureRequest {
            invocation: None,
            plan: &plan,
            selection_index: 0,
            phase: CapturePhase::Prefill,
            prediction: 0,
            projection: authority.producer(0).unwrap(),
            fragment_index: 0,
            producer_rank: 0,
        },
        &mut rejected,
    );
    let skipped = result.unwrap();
    assert!(matches!(
        skipped.record().outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::Limit {
                budget: CaptureBudget::Host,
                ..
            }
        }
    ));
    assert!(skipped.record().payload.is_none());
    assert_eq!(
        rejected.priced_values, 14,
        "summary requires selected terms, not local statistics"
    );
    assert_eq!(backend.transforms, 0);
    struct ScheduleSkip;
    impl CaptureReservation for ScheduleSkip {
        fn reserve(&mut self, _: CaptureUsage) -> Result<Option<CaptureSkipReason>, CaptureError> {
            Ok(Some(CaptureSkipReason::Schedule))
        }
    }
    let denied = capture_sum_fragment(
        &mut backend,
        &terms[0],
        PartitionCaptureRequest {
            invocation: None,
            plan: &plan,
            selection_index: 0,
            phase: CapturePhase::Prefill,
            prediction: 0,
            projection: authority.producer(0).unwrap(),
            fragment_index: 0,
            producer_rank: 0,
        },
        &mut ScheduleSkip,
    );
    assert!(
        denied.is_err(),
        "a skipped metadata reservation grants no host or native work"
    );
    assert_eq!(backend.transforms, 0);
    let integer = Value {
        shape: vec![3, 20],
        data: TensorObservationData::I64(vec![7; 60]),
    };
    assert!(capture_sum_fragment(
        &mut backend,
        &integer,
        PartitionCaptureRequest {
            invocation: None,
            plan: &plan,
            selection_index: 0,
            phase: CapturePhase::Prefill,
            prediction: 0,
            projection: authority.producer(0).unwrap(),
            fragment_index: 0,
            producer_rank: 0,
        },
        &mut ledger
    )
    .is_err());
    assert_eq!(backend.transforms, 0);
}

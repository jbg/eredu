use super::*;
use eredu_core::RoutingObservationField as Field;
fn routing_source(evidence: InterventionEvidence) -> (SharedCapturePlan, AdmittedInterventionPlan) {
    let (capture, _) = sources();
    let point = InterventionPoint {
        path: "router".into(),
        node_id: "block".into(),
        stage: InterventionStage::RoutingBeforeDispatch,
        axes: vec![
            TensorAxis {
                name: "token".into(),
                dimension: SymbolicDimension::TokenRows,
            },
            TensorAxis {
                name: "selected_expert".into(),
                dimension: SymbolicDimension::Known(2),
            },
        ],
        dtypes: vec![],
        operations: vec![InterventionKind::ExcludeExperts],
        score_stages: vec![],
        prefill: ObservationSupportStatus::Supported,
        decode: ObservationSupportStatus::Supported,
        conditions: vec![],
        routed_units: None,
        routing: Some(InterventionRoutingPolicy {
            expert_count: u32::MAX,
            top_k: 2,
            scoring: RoutingScoring::Softmax,
            normalize_selected: true,
            normalization_epsilon: 0.0,
            coefficient_scale: 1.0,
            groups: 1,
            selected_groups: 1,
            learned_coefficient_scale: false,
            shared_experts: 0,
        }),
    };
    let discovery = InterventionDiscovery {
        schema_version: 1,
        artifact_identity: "routing-source".into(),
        session_identity: Some("session".into()),
        points: vec![point],
    };
    let admitted = InterventionPlan {
        schema_version: 1,
        operations: vec![InterventionOperation {
            id: "routes".into(),
            target: "router".into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            action: InterventionAction::ExcludeExperts {
                expert_ids: vec![3],
            },
            evidence,
        }],
    }
    .admit(&discovery, capture.admission().request(), "session")
    .unwrap();
    (capture, admitted)
}
#[test]
fn routing_companion_has_four_ordered_paid_fields_and_preserves_unsigned_ids() {
    let (capture, admitted) = routing_source(InterventionEvidence::Preview { max_elements: 2 });
    let pool = capture_test_ledger(1 << 26, 0).unwrap();
    let source = pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&admitted).unwrap())
        .unwrap();
    let companion = source.plan().evidence(0).unwrap().geometry_source();
    let ids = CaptureTensorHostPlan::prepare(
        CaptureTensorGeometry::prepare(companion, 0, CapturePhase::Decode, 1, None).unwrap(),
    )
    .unwrap();
    let weights = CaptureTensorHostPlan::prepare(
        CaptureTensorGeometry::prepare(companion, 1, CapturePhase::Decode, 1, None).unwrap(),
    )
    .unwrap();
    assert_eq!(
        ids.retained_payload_bytes() - weights.retained_payload_bytes(),
        8,
        "two unsigned host values retain eight more bytes than two floating values"
    );
    let plan = CaptureRunHostPlan::prepare(&capture)
        .unwrap()
        .with_interventions(&source)
        .unwrap();
    let (reservation, run) = fresh(&pool, plan.initialization_peak_bytes());
    let mut bank = run.prepare_capture_run(&reservation, plan).unwrap();
    drop(
        bank.begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap(),
    );
    let mut frame = bank
        .begin_step(CapturePhase::Decode, 1)
        .unwrap()
        .prepare()
        .unwrap();
    let claim = frame.take_intervention(0).unwrap();
    frame
        .record_intervention(claim.finish(CaptureUsage::default()).unwrap())
        .unwrap();
    assert!(
        frame
            .take_intervention_evidence(0, InterventionEvidenceSide::Before)
            .is_err()
    );
    assert!(
        frame
            .take_routing_intervention_evidence(
                0,
                InterventionEvidenceSide::After,
                Field::SelectedExperts
            )
            .is_err()
    );
    let before = ledger(&pool);
    for side in [
        InterventionEvidenceSide::Before,
        InterventionEvidenceSide::After,
    ] {
        for field in [Field::SelectedExperts, Field::Coefficients] {
            let claim = frame
                .take_routing_intervention_evidence(0, side, field)
                .unwrap()
                .unwrap();
            assert_eq!(claim.routing_field(), Some(field));
            let (receipt, kind) = claim.into_parts();
            let CaptureInterventionEvidenceKind::Preview(claim) = kind else {
                panic!("preview")
            };
            let mut writer = claim.prepare().unwrap();
            assert_eq!(writer.len(), 2);
            let dtype = if field == Field::SelectedExperts {
                assert!(writer.push_f32(1.0).is_err());
                writer.push_u64(16_777_217).unwrap();
                writer.push_u64(u64::from(u32::MAX) - 1).unwrap();
                TensorDtype::U32
            } else {
                assert!(writer.push_u64(1).is_err());
                writer.push_f32(0.25).unwrap();
                writer.push_f32(0.75).unwrap();
                TensorDtype::F32
            };
            let evidence = receipt.finish_preview(writer.finish().unwrap()).unwrap();
            frame
                .record_intervention_evidence(evidence, dtype, CaptureUsage::default())
                .unwrap();
            assert!(
                frame
                    .take_routing_intervention_evidence(0, side, field)
                    .is_err()
            );
        }
    }
    assert_eq!(
        ledger(&pool),
        before,
        "field payloads consume the admitted account without an independent reservation"
    );
    let finished = frame
        .finish(CaptureStepOutcome::Committed, unlimited(), unlimited(), 0.0)
        .unwrap();
    let records = &finished.interventions[0].evidence;
    assert_eq!(records.len(), 4);
    for index in [0, 2] {
        let Some(CapturePayload::SharedTensor(values)) = &records[index].payload else {
            panic!("integer evidence")
        };
        assert_eq!(
            values.data(),
            &TensorObservationData::U64(vec![16_777_217, u64::from(u32::MAX) - 1])
        );
    }
}
#[test]
fn routing_frame_lends_all_four_claims_and_unmatched_is_a_completed_outcome() {
    let (capture, admitted) = routing_source(InterventionEvidence::Preview { max_elements: 2 });
    let pool = capture_test_ledger(1 << 26, 0).unwrap();
    let source = pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&admitted).unwrap())
        .unwrap();
    let plan = CaptureRunHostPlan::prepare(&capture)
        .unwrap()
        .with_interventions(&source)
        .unwrap();
    let (reservation, run) = fresh(&pool, plan.initialization_peak_bytes());
    let mut bank = run.prepare_capture_run(&reservation, plan).unwrap();
    drop(
        bank.begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap(),
    );
    let mut frame = bank
        .begin_step(CapturePhase::Decode, 1)
        .unwrap()
        .prepare()
        .unwrap();
    let claim = frame.take_intervention(0).unwrap();
    frame
        .record_intervention(claim.finish_unmatched(CaptureUsage::default()).unwrap())
        .unwrap();
    assert_eq!(
        frame.interventions()[0].outcome,
        InterventionOutcome::Unmatched
    );
    let mut loan = frame.take_intervention_evidence_frame(0).unwrap();
    assert_eq!(loan.frame().records().len(), 4);
    for index in 0..4 {
        drop(loan.frame_mut().take_tensor(index).unwrap());
    }
    frame.return_intervention_evidence_frame(loan).unwrap();
    let mut loan = frame.take_intervention_evidence_frame(0).unwrap();
    for index in 0..4 {
        assert!(loan.frame_mut().take_tensor(index).is_err());
    }
    drop(loan);
    assert!(
        frame
            .finish(CaptureStepOutcome::Committed, unlimited(), unlimited(), 0.0)
            .is_err(),
        "abandoned evidence never becomes a completed frame"
    );
}

#[test]
fn routing_prefill_retains_four_typed_targets_across_every_real_hook() {
    use crate::capture::{CapturePrefillHookDecision, CapturePrefillObservationPolicy};
    let (capture, admitted) = routing_source(InterventionEvidence::Preview { max_elements: 4 });
    let pool = capture_test_ledger(1 << 26, 0).unwrap();
    let source = pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&admitted).unwrap())
        .unwrap();
    let plan = CaptureRunHostPlan::prepare(&capture)
        .unwrap()
        .with_interventions(&source)
        .unwrap();
    let (reservation, run) = fresh(&pool, plan.initialization_peak_bytes());
    let mut bank = run.prepare_capture_run(&reservation, plan).unwrap();
    let mut frame = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let inference = InferenceGeometry {
        batch_size: 1,
        input_positions: 3,
        cached_positions: 0,
        max_output_tokens: 4,
        prefill_chunk_positions: 2,
        output: OutputDemand::Sequence,
    };
    let companion = source.plan().evidence(0).unwrap().shared_geometry_source();
    let policy = CapturePrefillObservationPolicy::new(companion, inference).unwrap();
    frame
        .prepare_prefill_intervention_evidence(inference)
        .unwrap();
    let claim = frame.take_intervention(0).unwrap();
    frame
        .record_intervention(claim.finish(CaptureUsage::default()).unwrap())
        .unwrap();
    let before = ledger(&pool);
    let mut quota = CaptureLedger::new(capture.admission());
    quota.begin_step();
    for ordinal in 0..2 {
        for index in 0..4 {
            let row = policy.row(index).unwrap();
            let fragment = row.assembly().unwrap().fragment(ordinal).unwrap();
            let chunk = crate::prefill::PrefillChunk {
                input: fragment.input().clone(),
                position: fragment.position(),
                output: fragment.output_demand(),
            };
            let mut loan = frame.take_intervention_evidence_frame(0).unwrap();
            let target = loan.frame_mut();
            let path = &companion.admission().plan().selections[index].path;
            let decision = target.begin_prefill_hook(index, &chunk, path).unwrap();
            if ordinal == 0 {
                assert_eq!(decision, CapturePrefillHookDecision::First);
                let dtype = if index % 2 == 0 {
                    TensorDtype::U32
                } else {
                    TensorDtype::F32
                };
                assert!(
                    target
                        .reserve_prefill_hook(
                            index,
                            &mut quota,
                            dtype,
                            CaptureUsage {
                                captures: 1,
                                ..CaptureUsage::default()
                            }
                        )
                        .unwrap()
                        .is_none()
                );
            }
            let mut writer = target
                .take_prefill_fragment(index, &fragment)
                .unwrap()
                .prepare()
                .unwrap();
            for mapping in fragment.mappings() {
                if index % 2 == 0 {
                    writer
                        .push_u64(16_777_217 + mapping.destination_index() as u64)
                        .unwrap();
                } else {
                    writer
                        .push_f32(0.125 * (mapping.destination_index() + 1) as f32)
                        .unwrap();
                }
            }
            writer.finish().unwrap();
            target.finish_prefill_hook(index, &fragment).unwrap();
            if index != 3 {
                assert!(
                    target.complete_prefill_chunk(ordinal).is_err(),
                    "later actual routing fields remain required"
                );
            }
            frame.return_intervention_evidence_frame(loan).unwrap();
        }
        let mut loan = frame.take_intervention_evidence_frame(0).unwrap();
        loan.frame_mut().complete_prefill_chunk(ordinal).unwrap();
        if ordinal == 1 {
            loan.frame_mut().finish_prefill_targets().unwrap();
        }
        frame.return_intervention_evidence_frame(loan).unwrap();
    }
    assert_eq!(ledger(&pool), before);
    let finished = frame
        .finish(CaptureStepOutcome::Committed, unlimited(), unlimited(), 0.0)
        .unwrap();
    for (index, record) in finished.interventions[0].evidence.iter().enumerate() {
        assert!(matches!(
            record.outcome,
            CaptureOutcome::Truncated {
                available_elements: 6,
                emitted_elements: 4
            }
        ));
        let Some(CapturePayload::SharedTensor(values)) = &record.payload else {
            panic!("typed field")
        };
        if index % 2 == 0 {
            assert_eq!(
                values.data(),
                &TensorObservationData::U64(vec![16_777_217, 16_777_218, 16_777_219, 16_777_220])
            );
        } else {
            assert_eq!(
                values.data(),
                &TensorObservationData::F32(vec![0.125, 0.25, 0.375, 0.5])
            );
        }
    }
}

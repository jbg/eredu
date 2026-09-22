use super::*;
use crate::intervention::{InterventionPrefillWindow, PreparedRoutedInterventionRows};
use crate::prefill::PrefillChunk;

fn sparse_sources() -> (SharedCapturePlan, AdmittedInterventionPlan) {
    let (capture, _) = sources();
    let discovery = InterventionDiscovery {
        schema_version: 1,
        artifact_identity: "sparse-source".into(),
        session_identity: Some("session".into()),
        points: vec![InterventionPoint {
            path: "units".into(),
            node_id: "block".into(),
            stage: InterventionStage::Activation,
            axes: vec![
                TensorAxis {
                    name: "token".into(),
                    dimension: SymbolicDimension::TokenRows,
                },
                TensorAxis {
                    name: "component".into(),
                    dimension: SymbolicDimension::Known(6),
                },
            ],
            dtypes: vec![InterventionDtype::Float32],
            operations: vec![InterventionKind::Scale],
            score_stages: vec![],
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            conditions: vec![],
            routing: None,
            routed_units: Some(RoutedUnitInterventionPoint {
                routing: "router".into(),
                geometry: RoutedUnitGeometry {
                    experts: 3,
                    units_per_expert: 2,
                    routes_per_token: 2,
                },
            }),
        }],
    };
    let edits = InterventionPlan {
        schema_version: 1,
        operations: vec![InterventionOperation {
            id: "scale".into(),
            target: "units".into(),
            schedule: Default::default(),
            slices: vec![],
            action: InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: -2.0,
            },
            evidence: InterventionEvidence::None,
        }],
    }
    .admit(&discovery, capture.admission().request(), "session")
    .unwrap();
    (capture, edits)
}
fn geometry() -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 3,
        max_output_tokens: 4,
        prefill_chunk_positions: 2,
        output: OutputDemand::Sequence,
    }
}
fn window(plan: &AdmittedInterventionPlan, index: u64) -> InterventionPrefillWindow {
    let start = index * 2;
    InterventionPrefillWindow::new(
        plan,
        geometry(),
        &PrefillChunk {
            input: start..(start + 2).min(3),
            position: start,
            output: OutputDemand::Sequence,
        },
    )
    .unwrap()
}

#[test]
fn sparse_prefill_keeps_one_claim_and_charge_across_native_batches_and_prompt_chunks() {
    let (capture, admitted) = sparse_sources();
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
    let execution = InferenceExecutionIdentity::default();
    let funding = pool
        .prepare_workspace_metadata(
            &execution,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, 1 << 26),
        )
        .unwrap();
    let usage = CaptureUsage {
        retained_bytes: 1024,
        host_bytes: 2048,
        ..Default::default()
    };
    let initial_charge = frame.interventions()[0].charged;
    for chunk in 0..2 {
        let window = window(source.plan().admission(), chunk);
        assert!(
            frame
                .finish_prefill_routed_intervention_chunk(0, window)
                .is_err()
        );
        for local in 0..window.physical().sequence {
            let mut cursor = frame
                .take_prefill_routed_intervention_cursor(0, window)
                .unwrap();
            assert!(cursor.begin_batch([local, local + 1]).is_err());
            if chunk == 0 && local == 0 {
                cursor.charge(usage).unwrap();
            } else {
                assert!(cursor.charge(usage).is_err());
            }
            let batch = cursor
                .begin_prefill_batch(window, [local, local + 1])
                .unwrap();
            let token = window.range()[0] + local;
            assert_eq!(batch.source_chunk(), (3, [token, token + 1]));
            assert_eq!(batch.prefill_window(), Some(window));
            let mut rows = PreparedRoutedInterventionRows::prepare_scheduled_prefill(
                &source,
                0,
                window,
                [local, local + 1],
                2,
                funding.clone(),
            )
            .unwrap();
            for slot in 0..2 {
                rows.push_row(RoutedUnitLocation {
                    source_peer: None,
                    token: local,
                    slot,
                    expert: slot,
                })
                .unwrap();
            }
            let prepared = rows.finish().unwrap();
            batch.finish(&prepared).unwrap();
            frame.retain_routed_intervention_cursor(cursor).unwrap();
        }
        frame
            .finish_prefill_routed_intervention_chunk(0, window)
            .unwrap();
        if window.is_final() {
            assert!(
                frame
                    .take_prefill_routed_intervention_cursor(0, window)
                    .is_err()
            );
        }
    }
    let receipt = &frame.interventions()[0];
    assert_eq!(receipt.outcome, InterventionOutcome::Applied);
    assert_eq!(receipt.charged, initial_charge.checked_add(usage).unwrap());
    assert_eq!(
        receipt.routed_units.unwrap(),
        RoutedUnitInterventionReceipt {
            source_tokens: 3,
            completed_tokens: 3,
            affected_values: 12,
        }
    );
    drop(frame);
    drop((bank, run, reservation, funding, source));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn abandoned_sparse_prefill_batch_cannot_reissue_claim_or_advance_prompt() {
    let (capture, admitted) = sparse_sources();
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
    let first = window(source.plan().admission(), 0);
    let mut cursor = frame
        .take_prefill_routed_intervention_cursor(0, first)
        .unwrap();
    cursor.charge(CaptureUsage::default()).unwrap();
    drop(cursor.begin_prefill_batch(first, [0, 1]).unwrap());
    assert!(cursor.begin_prefill_batch(first, [0, 1]).is_err());
    assert!(frame.retain_routed_intervention_cursor(cursor).is_err());
    assert!(
        frame
            .take_prefill_routed_intervention_cursor(0, first)
            .is_err()
    );
    assert!(
        frame
            .take_prefill_routed_intervention_cursor(0, window(source.plan().admission(), 1))
            .is_err()
    );
    assert!(
        frame
            .finish_prefill_routed_intervention_chunk(0, first)
            .is_err()
    );
    drop(frame);
    drop((bank, run, reservation, source));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

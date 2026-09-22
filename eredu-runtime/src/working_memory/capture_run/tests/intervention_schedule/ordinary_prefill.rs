//! The ordinary source initializes the same paid companion targets as delivery.
//! These tests exercise Host writers and cursor ownership, not native execution.
use super::*;
use crate::{
    capture::{CapturePrefillHookDecision, CapturePrefillObservationPolicy},
    intervention::InterventionPrefillWindow,
    prefill::PrefillChunk,
};

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
fn chunk(index: u64) -> PrefillChunk {
    let start = index * 2;
    PrefillChunk {
        input: start..(start + 2).min(3),
        position: start,
        output: OutputDemand::Sequence,
    }
}
fn preview_usage() -> CaptureUsage {
    CaptureUsage {
        captures: 1,
        host_bytes: 16,
        retained_bytes: 16,
        encoded_bytes: 4096,
    }
}
fn nonzero(index: usize) -> f32 {
    (index + 1) as f32 * if index % 2 == 0 { 1.0 } else { -1.0 }
}

#[test]
fn ordinary_prefill_preview_keeps_one_full_charge_per_side_and_independent_payloads() {
    let (capture, admitted) =
        sources_with_evidence(true, InterventionEvidence::Preview { max_elements: 4 });
    let pool = capture_test_ledger(1 << 26, 0).unwrap();
    let source = pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&admitted).unwrap())
        .unwrap();
    let plan = CaptureRunHostPlan::prepare(&capture)
        .unwrap()
        .with_interventions(&source)
        .unwrap();
    let (reservation, run) = fresh(&pool, plan.initialization_peak_bytes());
    let native = run.scope().unwrap();
    let mut bank = run.prepare_capture_run(&reservation, plan).unwrap();
    let mut frame = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare_prefill_with_progression(geometry())
        .unwrap();
    let physical = ledger(&pool);
    let mut quota = CaptureLedger::new(capture.admission());
    quota.begin_step();
    source.reserve_capture_metadata(&mut quota).unwrap();
    let metadata = quota.total();
    let mut payloads = Vec::new();

    for ordinal in 0..2 {
        let chunk = chunk(ordinal);
        let window =
            InterventionPrefillWindow::new(source.plan().admission(), geometry(), &chunk).unwrap();
        for operation in 0..2 {
            let mut cursor = frame.take_prefill_intervention(operation, window).unwrap();
            let fragment = cursor.begin(window).unwrap();
            fragment.claim().validate_native_custody(&native).unwrap();
            let mut loan = frame.take_intervention_evidence_frame(operation).unwrap();
            assert_eq!(loan.frame().prefill_geometry(), Some(geometry()));
            let companion = source
                .plan()
                .evidence(operation)
                .unwrap()
                .shared_geometry_source();
            assert!(loan.source().same_storage(companion));
            let policy = CapturePrefillObservationPolicy::new(companion, geometry()).unwrap();
            for side in 0..2 {
                let row = policy.row(side).unwrap();
                let fragment = row.assembly().unwrap().fragment(ordinal).unwrap();
                let path = &companion.admission().plan().selections[side].path;
                let target = loan.frame_mut();
                assert_eq!(
                    target.begin_prefill_hook(side, &chunk, path).unwrap(),
                    if ordinal == 0 {
                        CapturePrefillHookDecision::First
                    } else {
                        CapturePrefillHookDecision::Continue
                    }
                );
                if ordinal == 0 {
                    assert!(
                        target
                            .reserve_prefill_hook(
                                side,
                                &mut quota,
                                TensorDtype::F32,
                                preview_usage()
                            )
                            .unwrap()
                            .is_none()
                    );
                    let mut writer = target
                        .take_prefill_fragment(side, &fragment)
                        .unwrap()
                        .prepare()
                        .unwrap();
                    for mapping in fragment.mappings() {
                        // Operation two observes the first operation's doubled result.
                        let value =
                            nonzero(mapping.source_index()) * (1u32 << (operation + side)) as f32;
                        writer.push_f32(value).unwrap();
                    }
                    writer.finish().unwrap();
                } else {
                    assert_eq!(fragment.output_elements(), 0);
                }
                target.finish_prefill_hook(side, &fragment).unwrap();
            }
            loan.frame_mut().complete_prefill_chunk(ordinal).unwrap();
            if ordinal == 1 {
                loan.frame_mut().finish_prefill_targets().unwrap();
                for (side, record) in loan.frame().records().iter().enumerate() {
                    assert_eq!(record.charged.captures, 1);
                    assert_eq!(record.charged.retained_bytes, 16);
                    let Some(CapturePayload::SharedTensor(value)) = &record.payload else {
                        panic!("the original paid companion owns its payload");
                    };
                    let TensorObservationData::F32(values) = value.data() else {
                        panic!("the original floating writer preserves values");
                    };
                    let factor = (1u32 << (operation + side)) as f32;
                    assert_eq!(
                        &values[..],
                        &[factor, -2.0 * factor, 3.0 * factor, -4.0 * factor]
                    );
                    payloads.push(value.clone());
                }
            }
            frame.return_intervention_evidence_frame(loan).unwrap();
            fragment.finish().unwrap();
            frame.retain_prefill_intervention(cursor).unwrap();
        }
        frame.validate_prefill_intervention_end(window).unwrap();
        frame.complete_prefill_chunk(ordinal).unwrap();
        assert_eq!(
            quota.total(),
            metadata
                .checked_add(preview_usage().checked_mul(4).unwrap())
                .unwrap()
        );
        assert_eq!(
            ledger(&pool),
            physical,
            "both uneven chunks use the same paid targets"
        );
    }
    frame.finish_prefill_targets().unwrap();
    let delivered = finish(frame);
    assert!(
        delivered
            .interventions()
            .iter()
            .all(|r| r.outcome == InterventionOutcome::Applied)
    );
    let after = payloads.pop().unwrap();
    let before = payloads.pop().unwrap();
    let (TensorObservationData::F32(a), TensorObservationData::F32(b)) =
        (before.data(), after.data())
    else {
        unreachable!()
    };
    assert_ne!(
        a.as_ptr(),
        b.as_ptr(),
        "Before and After retain distinct backing"
    );
    drop((payloads, delivered, bank, reservation, run, source));
    native.certify().unwrap();
    assert!(pool.payload_used_bytes().unwrap() > 0);
    drop(before);
    assert!(
        pool.payload_used_bytes().unwrap() > 0,
        "After has independent custody"
    );
    assert!(
        matches!(after.data(), TensorObservationData::F32(values) if values == &[4., -8., 12., -16.])
    );
    drop(after);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn abandoned_ordinary_prefill_evidence_keeps_spent_quota_and_cannot_reissue_the_frame() {
    let (capture, admitted) =
        sources_with_evidence(true, InterventionEvidence::Preview { max_elements: 4 });
    let pool = capture_test_ledger(1 << 26, 0).unwrap();
    let source = pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&admitted).unwrap())
        .unwrap();
    let plan = CaptureRunHostPlan::prepare(&capture)
        .unwrap()
        .with_interventions(&source)
        .unwrap();
    let (reservation, run) = fresh(&pool, plan.initialization_peak_bytes());
    let native = run.scope().unwrap();
    let mut bank = run.prepare_capture_run(&reservation, plan).unwrap();
    let mut frame = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare_prefill_with_progression(geometry())
        .unwrap();
    let window =
        InterventionPrefillWindow::new(source.plan().admission(), geometry(), &chunk(0)).unwrap();
    let mut cursor = frame.take_prefill_intervention(0, window).unwrap();
    let intervention = cursor.begin(window).unwrap();
    intervention
        .claim()
        .validate_native_custody(&native)
        .unwrap();
    let companion = source.plan().evidence(0).unwrap().shared_geometry_source();
    let policy = CapturePrefillObservationPolicy::new(companion, geometry()).unwrap();
    let row = policy.row(0).unwrap();
    let fragment = row.assembly().unwrap().fragment(0).unwrap();
    let mut quota = CaptureLedger::new(capture.admission());
    quota.begin_step();
    let mut loan = frame.take_intervention_evidence_frame(0).unwrap();
    assert_eq!(loan.frame().prefill_geometry(), Some(geometry()));
    let target = loan.frame_mut();
    target
        .begin_prefill_hook(
            0,
            &chunk(0),
            &companion.admission().plan().selections[0].path,
        )
        .unwrap();
    target
        .reserve_prefill_hook(0, &mut quota, TensorDtype::F32, preview_usage())
        .unwrap();
    {
        let mut writer = target
            .take_prefill_fragment(0, &fragment)
            .unwrap()
            .prepare()
            .unwrap();
        writer.push_f32(7.0).unwrap();
    }
    assert!(target.complete_prefill_chunk(0).is_err());
    let spent = quota.total();
    drop(loan);
    drop(intervention);
    assert!(cursor.begin(window).is_err());
    drop(cursor);
    assert!(frame.take_intervention_evidence_frame(0).is_err());
    assert!(frame.take_prefill_intervention(0, window).is_err());
    assert!(frame.validate_prefill_intervention_end(window).is_err());
    assert_eq!(quota.total(), spent);
    assert_eq!(spent, preview_usage());
    drop(frame);
    drop((bank, reservation, run, source));
    native.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

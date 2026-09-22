use super::candidates::{g, source_with_transform, value_usage};
use super::*;
use crate::capture::{
    CaptureObservationStep, CapturePrefillHookDecision as D, CapturePrefillObservationPolicy,
};
use crate::prefill::PrefillChunk;
fn source(prompt: u64) -> SharedCapturePlan {
    source_with_transform(
        prompt,
        CaptureTransform::TokenScores {
            token_ids: vec![2, 0],
        },
        CaptureLimitPolicy::Fail,
        u64::MAX,
    )
}
fn score(id: u32) -> CaptureTokenScore {
    // Full distribution [0,1,2,3]; the two selected IDs are deliberately unsorted.
    let z = (0..4).map(|i| (i as f64).exp()).sum::<f64>().ln();
    CaptureTokenScore {
        target: CaptureCandidate {
            token_id: id,
            score: id as f32,
            allowed: true,
        },
        log_probability: id as f64 - z,
        rank: 4 - id as u64,
        strongest_alternative: Some(CaptureCandidate {
            token_id: 3,
            score: 3.,
            allowed: true,
        }),
    }
}
fn fill(claim: CaptureTokenScoreClaim<'_, '_>) -> ClaimedCaptureTokenScores {
    let mut builder = claim.prepare(None).unwrap();
    builder.push(score(2)).unwrap();
    builder.push(score(0)).unwrap();
    builder
        .finish((0..4).map(|i| (i as f64).exp()).sum::<f64>().ln())
        .unwrap()
}
#[test]
fn selected_scores_use_only_terminal_chunk_and_escape_with_original_host_custody() {
    for prompt in [5, 6] {
        for output in [OutputDemand::LastPosition, OutputDemand::Sequence] {
            let source = source(prompt);
            assert!(
                !CaptureObservationStep::new(source.admission(), CapturePhase::Prefill, 0)
                    .unwrap()
                    .requires_sequence_readout()
            );
            let h = plan(&source).initialization_peak_bytes();
            let pool = capture_test_ledger(h, 0).unwrap();
            let (reservation, run) = fresh(&pool, h);
            let mut bank = run
                .prepare_capture_run(&reservation, plan(&source))
                .unwrap();
            let geometry = g(prompt, output);
            let mut step = bank
                .begin_step(CapturePhase::Prefill, 0)
                .unwrap()
                .prepare_prefill_with_progression(geometry)
                .unwrap();
            let policy = CapturePrefillObservationPolicy::new(&source, geometry).unwrap();
            let row = policy.row(0).unwrap();
            let mut cold = row.initial_progress();
            let mut quota = CaptureLedger::new(source.admission());
            quota.begin_step();
            CaptureObservationStep::new(source.admission(), CapturePhase::Prefill, 0)
                .unwrap()
                .reserve_metadata(&mut quota)
                .unwrap();
            let mut pointer = std::ptr::null();
            for i in 0..3 {
                let end = ((i + 1) * 2).min(prompt);
                let chunk = PrefillChunk {
                    input: i * 2..end,
                    position: 2 + i * 2,
                    output: output.for_chunk(end == prompt),
                };
                let decision = step
                    .begin_prefill_hook(0, &chunk, MODEL_LOGITS_OBSERVATION_PATH)
                    .unwrap();
                assert_eq!(
                    decision,
                    row.begin_hook(&mut cold, &chunk, MODEL_LOGITS_OBSERVATION_PATH)
                        .unwrap()
                );
                if end != prompt {
                    assert_eq!(decision, D::Ignore);
                    assert!(step.take_token_scores(0).is_err());
                } else {
                    assert_eq!(decision, D::First);
                    step.reserve_prefill_hook(0, &mut quota, TensorDtype::F32, value_usage())
                        .unwrap();
                    let claim = step.take_token_scores(0).unwrap();
                    assert_eq!(
                        claim.geometry().source_shape()[1],
                        if output == OutputDemand::Sequence {
                            (end - i * 2) as usize
                        } else {
                            1
                        }
                    );
                    let receipt = fill(claim);
                    pointer = receipt.observation().scores.as_ptr();
                    step.record_token_scores(receipt, TensorDtype::F32, CaptureUsage::default())
                        .unwrap();
                    step.finish_token_scores_prefill_hook(0, &chunk).unwrap();
                    let mut cold_quota = CaptureLedger::new(source.admission());
                    cold_quota.begin_step();
                    row.reserve_first(&mut cold, &mut cold_quota, value_usage())
                        .unwrap();
                    row.finish_token_scores_hook(&mut cold, &chunk).unwrap();
                    assert!(step.take_token_scores(0).is_err());
                }
                step.complete_prefill_chunk(i).unwrap();
                row.advance_chunk(&mut cold, i).unwrap();
            }
            step.finish_prefill_targets().unwrap();
            let frame = step
                .prepare_delivery(quota.step(), quota.total(), 0.)
                .unwrap()
                .finish(CaptureStepOutcome::Committed);
            let alias = frame.clone();
            let Some(CapturePayload::TokenScores(values)) = &frame.records()[0].payload else {
                panic!("scores")
            };
            assert_eq!(values.scores.as_ptr(), pointer);
            assert_eq!(values.scores[0].target.token_id, 2);
            assert_eq!(frame.step_usage().captures, 1);
            drop(frame);
            drop((bank, reservation, run));
            assert_eq!(pool.payload_used_bytes().unwrap(), h);
            drop(alias);
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        }
    }
}
#[test]
fn ordered_score_claim_rejects_foreign_frame_and_poisoned_partial_fill_without_refund() {
    let source = source(3);
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h * 2, 0).unwrap();
    let (r, run) = fresh(&pool, h * 2);
    let mut a = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut b = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut sa = a
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let mut sb = b
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let receipt = fill(sa.take_token_scores(0).unwrap());
    assert!(matches!(
        sb.record_token_scores(receipt, TensorDtype::F32, value_usage()),
        Err(CaptureRunHostError::ReceiptMismatch)
    ));
    assert!(sb.records()[0].payload.is_none());
    assert!(sa.take_token_scores(0).is_err());
    let mut partial = sb.take_token_scores(0).unwrap().prepare(None).unwrap();
    partial.push(score(2)).unwrap();
    assert!(partial.push(score(2)).is_err());
    assert!(partial.push(score(0)).is_err());
    let failure = partial.finish(4.).unwrap_err();
    assert!(sb.take_token_scores(0).is_err());
    drop((sa, sb));
    drop((a, b, r, run));
    assert_eq!(pool.payload_used_bytes().unwrap(), h);
    drop(failure);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

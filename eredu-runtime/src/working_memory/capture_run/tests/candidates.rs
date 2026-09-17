//! Fixed original host ownership and source-bound terminal progression.
use super::*;
use crate::capture::{
    CaptureObservationStep, CapturePrefillHookDecision as D, CapturePrefillObservationPolicy,
};
use crate::prefill::PrefillChunk;
fn source(prompt: u64, count: u64) -> SharedCapturePlan {
    source_with_limit(prompt, count, CaptureLimitPolicy::Fail, u64::MAX)
}
fn source_with_limit(
    prompt: u64,
    count: u64,
    policy: CaptureLimitPolicy,
    captures: u64,
) -> SharedCapturePlan {
    source_with_transform(
        prompt,
        CaptureTransform::TopCandidates { count },
        policy,
        captures,
    )
}
pub(super) fn source_with_transform(
    prompt: u64,
    transform: CaptureTransform,
    policy: CaptureLimitPolicy,
    captures: u64,
) -> SharedCapturePlan {
    let mut p = point();
    p.path = MODEL_LOGITS_OBSERVATION_PATH.into();
    p.axes = Some(vec![
        TensorAxis {
            name: "batch".into(),
            dimension: SymbolicDimension::Batch,
        },
        TensorAxis {
            name: "sequence".into(),
            dimension: SymbolicDimension::Sequence,
        },
        TensorAxis {
            name: "vocabulary".into(),
            dimension: SymbolicDimension::Known(4),
        },
    ]);
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![p],
        completeness: DescriptionCompleteness::Complete,
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: MODEL_LOGITS_OBSERVATION_PATH.into(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let caps = CaptureCapabilities {
        transformations: vec![
            CaptureTransformKind::TopCandidates,
            CaptureTransformKind::TokenScores,
        ],
        max_histogram_bins: 0,
        physical_native_limit: false,
        conditions: vec![],
    };
    let mut raw = CapturePlan::none();
    raw.selections.push(CaptureSelection {
        id: "candidates".into(),
        path: MODEL_LOGITS_OBSERVATION_PATH.into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        transform,
    });
    raw.limits.per_step = unlimited();
    raw.limits.cumulative = unlimited();
    raw.limits.per_step.captures = captures;
    raw.limits.on_limit = policy;
    SharedCapturePlan::new(
        raw.admit_with_text_origin(
            &catalog,
            &support,
            &caps,
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: prompt,
                max_predictions: 4,
            },
            CaptureTextOrigin {
                cached_positions: 2,
            },
        )
        .unwrap(),
    )
}
pub(super) fn g(prompt: u64, output: OutputDemand) -> InferenceGeometry {
    InferenceGeometry {
        batch_size: 1,
        cached_positions: 2,
        input_positions: prompt,
        max_output_tokens: 4,
        prefill_chunk_positions: 2,
        output,
    }
}
pub(super) fn value_usage() -> CaptureUsage {
    CaptureUsage {
        captures: 1,
        retained_bytes: 128,
        host_bytes: 128,
        encoded_bytes: 8192,
    }
}
fn fill(claim: CaptureCandidateClaim<'_, '_>) -> ClaimedCaptureCandidates {
    let mut output = claim.prepare(None).unwrap();
    output.push(3, 7.0, true).unwrap();
    output.push(1, -0.25, true).unwrap();
    output.finish().unwrap()
}
#[test]
fn candidates_are_one_terminal_receipt_with_original_h_and_no_earlier_claim() {
    for prompt in [5, 6] {
        for output in [OutputDemand::LastPosition, OutputDemand::Sequence] {
            let source = source(prompt, 2);
            let geometry = g(prompt, output);
            assert!(
                !CaptureObservationStep::new(source.admission(), CapturePhase::Prefill, 0)
                    .unwrap()
                    .requires_sequence_readout()
            );
            let h = plan(&source).initialization_peak_bytes();
            let pool = WorkingMemoryPool::new(h, 0).unwrap();
            let (reservation, run) = fresh(&pool, h);
            let mut bank = run
                .prepare_capture_run(&reservation, plan(&source))
                .unwrap();
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
            let metadata = quota.total();
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
                    assert!(step.take_candidates(0).is_err());
                    assert_eq!(quota.total(), metadata);
                    assert!(step.records()[0].payload.is_none());
                } else {
                    assert_eq!(decision, D::First);
                    let mut cold_quota = CaptureLedger::new(source.admission());
                    cold_quota.begin_step();
                    row.reserve_first(&mut cold, &mut cold_quota, value_usage())
                        .unwrap();
                    step.reserve_prefill_hook(0, &mut quota, TensorDtype::F32, value_usage())
                        .unwrap();
                    let claim = step.take_candidates(0).unwrap();
                    assert_eq!(
                        claim.geometry().source_shape()[1],
                        if output == OutputDemand::Sequence {
                            (end - i * 2) as usize
                        } else {
                            1
                        }
                    );
                    let receipt = fill(claim);
                    pointer = receipt.observation().candidates.as_ptr();
                    step.record_candidates(receipt, TensorDtype::F32, CaptureUsage::default())
                        .unwrap();
                    step.finish_candidate_prefill_hook(0, &chunk).unwrap();
                    row.finish_candidate_hook(&mut cold, &chunk).unwrap();
                    assert!(step.take_candidates(0).is_err());
                }
                step.complete_prefill_chunk(i).unwrap();
                row.advance_chunk(&mut cold, i).unwrap();
            }
            step.finish_prefill_targets().unwrap();
            let delivery = step
                .prepare_delivery(quota.step(), quota.total(), 0.0)
                .unwrap();
            let frame = delivery.finish(CaptureStepOutcome::Committed);
            let alias = frame.clone();
            let Some(CapturePayload::Candidates(values)) = &frame.records()[0].payload else {
                panic!("candidate result")
            };
            assert_eq!(values.candidates.as_ptr(), pointer);
            assert_eq!(values.candidates[0].token_id, 3);
            assert_eq!(frame.step_usage().captures, 1);
            assert!(cold.finished());
            drop(frame);
            drop((bank, reservation, run));
            assert_eq!(pool.used_bytes().unwrap(), h);
            drop(alias);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}
#[test]
fn candidate_original_exact_short_foreign_receipt_and_partial_failure_do_not_refill() {
    let source = source(3, 2);
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h * 2, 0).unwrap();
    let (short, short_run) = fresh(&pool, h - 1);
    let before = CLAIM_ALLOCATIONS.get();
    assert!(
        short_run
            .prepare_capture_run(&short, plan(&source))
            .is_err()
    );
    assert_eq!(CLAIM_ALLOCATIONS.get(), before);
    drop((short, short_run));
    let (reservation, run) = fresh(&pool, h * 2);
    let mut a = run
        .prepare_capture_run(&reservation, plan(&source))
        .unwrap();
    let mut b = run
        .prepare_capture_run(&reservation, plan(&source))
        .unwrap();
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
    let receipt = fill(sa.take_candidates(0).unwrap());
    assert!(matches!(
        sb.record_candidates(receipt, TensorDtype::F32, value_usage()),
        Err(CaptureRunHostError::ReceiptMismatch)
    ));
    assert!(sb.records()[0].payload.is_none());
    assert!(sa.take_candidates(0).is_err());
    let mut partial = sb.take_candidates(0).unwrap().prepare(None).unwrap();
    partial.push(2, 3.0, true).unwrap();
    let error = partial.finish().unwrap_err();
    assert!(sb.take_candidates(0).is_err());
    drop((sa, sb));
    drop((a, b, reservation, run));
    assert_eq!(pool.used_bytes().unwrap(), h);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn missing_terminal_and_late_abort_never_finish_the_candidate_progression_twice() {
    for emit in [false, true] {
        let source = source(5, 2);
        let h = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
        let mut step = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare_prefill_with_progression(g(5, OutputDemand::LastPosition))
            .unwrap();
        for i in 0..2 {
            step.complete_prefill_chunk(i).unwrap();
        }
        if emit {
            let chunk = PrefillChunk {
                input: 4..5,
                position: 6,
                output: OutputDemand::LastPosition,
            };
            step.begin_prefill_hook(0, &chunk, MODEL_LOGITS_OBSERVATION_PATH)
                .unwrap();
            let mut quota = CaptureLedger::new(source.admission());
            quota.begin_step();
            CaptureObservationStep::new(source.admission(), CapturePhase::Prefill, 0)
                .unwrap()
                .reserve_metadata(&mut quota)
                .unwrap();
            step.reserve_prefill_hook(0, &mut quota, TensorDtype::F32, value_usage())
                .unwrap();
            let receipt = fill(step.take_candidates(0).unwrap());
            step.record_candidates(receipt, TensorDtype::F32, CaptureUsage::default())
                .unwrap();
            step.finish_candidate_prefill_hook(0, &chunk).unwrap();
            step.complete_prefill_chunk(2).unwrap();
            step.finish_prefill_targets().unwrap();
            let pending = step
                .prepare_delivery(quota.step(), quota.total(), 0.0)
                .unwrap();
            let frame = pending.finish(CaptureStepOutcome::Aborted);
            assert_eq!(frame.outcome(), CaptureStepOutcome::Aborted);
            assert_eq!(frame.step_usage().captures, 1);
        } else {
            assert!(step.complete_prefill_chunk(2).is_err());
            drop(step);
        }
        drop((bank, r, run));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn candidate_skip_fail_and_bad_fill_keep_terminal_attempt_once_only() {
    for policy in [CaptureLimitPolicy::Skip, CaptureLimitPolicy::Fail] {
        let source = source_with_limit(6, 2, policy, 0);
        let progression =
            CapturePrefillObservationPolicy::new(&source, g(6, OutputDemand::Sequence)).unwrap();
        let row = progression.row(0).unwrap();
        let mut state = row.initial_progress();
        for i in 0..2 {
            row.advance_chunk(&mut state, i).unwrap();
        }
        let chunk = PrefillChunk {
            input: 4..6,
            position: 6,
            output: OutputDemand::Sequence,
        };
        assert_eq!(
            row.begin_hook(&mut state, &chunk, MODEL_LOGITS_OBSERVATION_PATH)
                .unwrap(),
            D::First
        );
        let mut ledger = CaptureLedger::new(source.admission());
        ledger.begin_step();
        let result = row.reserve_first(&mut state, &mut ledger, value_usage());
        if policy == CaptureLimitPolicy::Skip {
            assert!(result.unwrap().is_some());
            row.advance_chunk(&mut state, 2).unwrap();
            assert!(state.finished());
        } else {
            assert!(result.is_err());
            assert!(row.advance_chunk(&mut state, 2).is_err());
        }
        assert!(
            row.reserve_first(&mut state, &mut ledger, value_usage())
                .is_err()
        );
    }
    let source = source(3, 2);
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let mut partial = step.take_candidates(0).unwrap().prepare(None).unwrap();
    partial.push(1, 2.0, true).unwrap();
    assert!(partial.push(2, 3.0, true).is_err());
    assert!(partial.push(0, 1.0, true).is_err());
    let error = partial.finish().unwrap_err();
    drop(step);
    drop((bank, r, run));
    assert_eq!(pool.used_bytes().unwrap(), h);
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

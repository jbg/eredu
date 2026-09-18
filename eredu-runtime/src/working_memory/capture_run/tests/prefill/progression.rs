//! Private activation fixtures prove logical/host behavior, not causal semantics.
use super::*;
use crate::capture::{
    CaptureObservationStep, CapturePrefillHookDecision as D, CapturePrefillObservationPolicy,
    CapturePrefillProgressError as E, CaptureProtocolError,
};
use crate::prefill::PrefillChunk;
use eredu_nn::{BlockFp8InputReconstructionPlan, GeneratedTensorProgram};
fn chunk(fragment: &CapturePrefillFragment<'_, '_>) -> PrefillChunk {
    PrefillChunk {
        input: fragment.input().clone(),
        position: fragment.position(),
        output: fragment.output_demand(),
    }
}
fn g() -> InferenceGeometry {
    InferenceGeometry {
        prefill_chunk_positions: 2,
        ..geometry()
    }
}
fn logical(source: &SharedCapturePlan) -> CaptureLedger {
    let mut ledger = CaptureLedger::new(source.admission());
    ledger.begin_step();
    CaptureObservationStep::new(source.admission(), CapturePhase::Prefill, 0)
        .unwrap()
        .reserve_metadata(&mut ledger)
        .unwrap();
    ledger
}
fn host_error(error: CaptureRunHostError) -> E {
    match error {
        CaptureRunHostError::Prefill(CapturePrefillHostError::Progression(error)) => error,
        _ => panic!("expected typed progression rejection: {error:?}"),
    }
}

#[test]
fn shared_cold_and_host_progression_charge_full_rows_once_and_keep_one_scatter_buffer() {
    let source = rows(5);
    let policy = CapturePrefillObservationPolicy::new(&source, g()).unwrap();
    let rows = [policy.row(0).unwrap(), policy.row(1).unwrap()];
    let mut cold = [rows[0].initial_progress(), rows[1].initial_progress()];
    let mut cold_ledger = logical(&source);
    let metadata = cold_ledger.total();
    let mut host_ledger = logical(&source);
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare_prefill_with_progression(g())
        .unwrap();
    let mut pointers = [std::ptr::null::<f32>(); 2];
    for k in 0..2 {
        for i in 0..2 {
            let fragment = rows[i].assembly().unwrap().fragment(k).unwrap();
            let actual = chunk(&fragment);
            let expected = if k == 0 { D::First } else { D::Continue };
            assert_eq!(
                rows[i]
                    .begin_hook(&mut cold[i], &actual, "unrelated")
                    .unwrap(),
                D::Ignore
            );
            assert_eq!(
                rows[i]
                    .begin_hook(&mut cold[i], &actual, "block.output")
                    .unwrap(),
                expected
            );
            assert_eq!(
                step.begin_prefill_hook(i, &actual, "block.output").unwrap(),
                expected
            );
            if k == 0 {
                assert!(rows[i]
                    .reserve_first(&mut cold[i], &mut cold_ledger, charge())
                    .unwrap()
                    .is_none());
                assert!(step
                    .reserve_prefill_hook(i, &mut host_ledger, TensorDtype::F32, charge())
                    .unwrap()
                    .is_none());
            }
            write_fragment(&mut step, i, rows[i].assembly().unwrap(), k);
            rows[i].finish_hook(&mut cold[i], &fragment).unwrap();
            step.finish_prefill_hook(i, &fragment).unwrap();
            let pointer = step.prefill_storage(i).unwrap().0;
            if k == 0 {
                pointers[i] = pointer;
            } else {
                assert_eq!(pointers[i], pointer);
            }
        }
        for i in 0..2 {
            rows[i].validate_chunk_end(&cold[i], k).unwrap();
        }
        step.complete_prefill_chunk(k).unwrap();
        for i in 0..2 {
            rows[i].advance_chunk(&mut cold[i], k).unwrap();
        }
    }
    assert!(cold.iter().all(|p| p.finished()));
    assert_eq!(host_ledger.total(), cold_ledger.total());
    assert_eq!(
        cold_ledger.total(),
        metadata
            .checked_add(charge().checked_mul(2).unwrap())
            .unwrap()
    );
    step.finish_prefill_targets().unwrap();
    assert_eq!(
        values(tensor(&step.records()[0])),
        &[0., 1., 10., 11., 20., 21., 100., 101., 110., 111., 120., 121.]
    );
    assert_eq!(values(tensor(&step.records()[1])), &[0., 1., 10., 11., 20.]);
    let escaped = tensor(&step.records()[0]).clone();
    drop(step);
    drop((bank, r, run));
    assert_eq!(pool.used_bytes().unwrap(), h);
    drop(escaped);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn missing_zero_hook_blocks_whole_row_advance_and_duplicate_zero_is_rejected() {
    let source = rows(0);
    let policy = CapturePrefillObservationPolicy::new(&source, geometry()).unwrap();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare_prefill_with_progression(geometry())
        .unwrap();
    let mut ledger = logical(&source);
    for k in 0..3 {
        let a = policy.row(0).unwrap();
        let b = policy.row(1).unwrap();
        let fa = a.assembly().unwrap().fragment(k).unwrap();
        let fb = b.assembly().unwrap().fragment(k).unwrap();
        let c = chunk(&fa);
        step.begin_prefill_hook(0, &c, "block.output").unwrap();
        if k == 0 {
            step.reserve_prefill_hook(0, &mut ledger, TensorDtype::F32, charge())
                .unwrap();
        }
        write_fragment(&mut step, 0, a.assembly().unwrap(), k);
        step.finish_prefill_hook(0, &fa).unwrap();
        assert!(matches!(
            host_error(step.complete_prefill_chunk(k).unwrap_err()),
            E::Missing
        ));
        assert!(
            matches!(
                host_error(step.begin_prefill_hook(0, &c, "block.output").unwrap_err()),
                E::Policy(CaptureProtocolError::Duplicate)
            ),
            "earlier row did not advance after later-row failure"
        );
        step.begin_prefill_hook(1, &c, "block.output").unwrap();
        assert_eq!(fb.output_elements(), 0);
        if k == 0 {
            step.reserve_prefill_hook(1, &mut ledger, TensorDtype::F32, charge())
                .unwrap();
            write_fragment(&mut step, 1, b.assembly().unwrap(), k); // one genuine empty target
        }
        step.finish_prefill_hook(1, &fb).unwrap();
        assert!(matches!(
            host_error(step.begin_prefill_hook(1, &c, "block.output").unwrap_err()),
            E::Policy(CaptureProtocolError::Duplicate)
        ));
        step.complete_prefill_chunk(k).unwrap();
    }
    step.finish_prefill_targets().unwrap();
    assert!(values(tensor(&step.records()[1])).is_empty());
    drop(step);
    drop((bank, r, run));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn exact_source_and_schedule_bind_progress_without_mutating_old_decode_policy() {
    let source = rows(5);
    let equal = SharedCapturePlan::new(source.admission().clone());
    let clone = source.clone();
    assert_eq!(source.admission().identity(), equal.admission().identity());
    let policy = CapturePrefillObservationPolicy::new(&source, g()).unwrap();
    let row = policy.row(0).unwrap();
    let mut progress = row.initial_progress();
    let fragment = row.assembly().unwrap().fragment(0).unwrap();
    let mut actual = chunk(&fragment);
    let other = CapturePrefillObservationPolicy::new(&equal, g()).unwrap();
    assert!(matches!(
        other
            .row(0)
            .unwrap()
            .begin_hook(&mut progress, &actual, "block.output"),
        Err(E::Identity)
    ));
    actual.position += 1;
    assert!(matches!(
        row.begin_hook(&mut progress, &actual, "block.output"),
        Err(E::Order)
    ));
    actual = chunk(&fragment);
    assert_eq!(
        CapturePrefillObservationPolicy::new(&clone, g())
            .unwrap()
            .row(0)
            .unwrap()
            .begin_hook(&mut progress, &actual, "block.output")
            .unwrap(),
        D::First
    );
    let decode = CaptureObservationStep::new(source.admission(), CapturePhase::Decode, 1).unwrap();
    assert!(decode
        .select(
            0,
            crate::capture::CaptureRecordStatus::Missing,
            "block.output"
        )
        .unwrap());
    assert!(matches!(
        decode.select(
            0,
            crate::capture::CaptureRecordStatus::Consumed,
            "block.output"
        ),
        Err(CaptureProtocolError::Duplicate)
    ));
    assert!(!decode.requires_sequence_readout());
}

#[test]
fn abandoned_or_failed_attempt_cannot_retry_or_refund_its_logical_usage() {
    let source = rows(5);
    let policy = CapturePrefillObservationPolicy::new(&source, g()).unwrap();
    let row = policy.row(0).unwrap();
    let fragment = row.assembly().unwrap().fragment(0).unwrap();
    let actual = chunk(&fragment);
    let mut progress = row.initial_progress();
    let result = catch_unwind(AssertUnwindSafe(|| {
        row.begin_hook(&mut progress, &actual, "block.output")
            .unwrap();
        panic!("source validation unwound before logical reservation");
    }));
    assert!(result.is_err());
    assert!(matches!(
        row.begin_hook(&mut progress, &actual, "block.output"),
        Err(E::Attempt)
    ));
    assert!(matches!(
        row.validate_chunk_end(&progress, 0),
        Err(E::Attempt)
    ));
    let mut charged = row.initial_progress();
    let mut ledger = logical(&source);
    row.begin_hook(&mut charged, &actual, "block.output")
        .unwrap();
    row.reserve_first(&mut charged, &mut ledger, charge())
        .unwrap();
    let used = ledger.total();
    assert!(matches!(
        row.reserve_first(&mut charged, &mut ledger, charge()),
        Err(E::Attempt)
    ));
    charged.fail_hook();
    assert!(matches!(
        row.finish_hook(&mut charged, &fragment),
        Err(E::Attempt)
    ));
    assert!(matches!(
        row.validate_chunk_end(&charged, 0),
        Err(E::Attempt)
    ));
    assert_eq!(ledger.total(), used);
}

fn limited(skip: bool) -> SharedCapturePlan {
    let original = rows(5);
    let mut raw = original.admission().plan().clone();
    raw.limits.per_step.captures = 0;
    raw.limits.on_limit = if skip {
        CaptureLimitPolicy::Skip
    } else {
        CaptureLimitPolicy::Fail
    };
    admit(raw, original.admission().points()[0].clone(), 4, false)
}
#[test]
fn quota_skip_disables_later_hooks_and_fail_remains_terminal_without_new_counter() {
    for skip in [false, true] {
        let source = limited(skip);
        let policy = CapturePrefillObservationPolicy::new(&source, g()).unwrap();
        let row = policy.row(0).unwrap();
        let mut progress = row.initial_progress();
        let mut ledger = logical(&source);
        let before = ledger.total();
        let f = row.assembly().unwrap().fragment(0).unwrap();
        let c = chunk(&f);
        row.begin_hook(&mut progress, &c, "block.output").unwrap();
        let result = row.reserve_first(&mut progress, &mut ledger, charge());
        assert_eq!(ledger.total(), before);
        if skip {
            assert!(matches!(
                result.unwrap(),
                Some(CaptureSkipReason::Limit { .. })
            ));
            row.validate_chunk_end(&progress, 0).unwrap();
            progress.advance_preflighted();
            let next = row.assembly().unwrap().fragment(1).unwrap();
            assert_eq!(
                row.begin_hook(&mut progress, &chunk(&next), "block.output")
                    .unwrap(),
                D::Ignore
            );
            row.validate_chunk_end(&progress, 1).unwrap();
            progress.advance_preflighted();
            assert!(progress.finished());
        } else {
            assert!(matches!(result, Err(E::Quota(CaptureError::Limit { .. }))));
            assert!(matches!(
                row.begin_hook(&mut progress, &c, "block.output"),
                Err(E::Attempt)
            ));
        }
    }
}

#[test]
fn generated_full_padded_creation_charged_once_while_preview_zero_preserves_factories() {
    let mut p = point();
    p.axes = Some(vec![
        TensorAxis {
            name: "head".into(),
            dimension: SymbolicDimension::Known(2),
        },
        TensorAxis {
            name: "row".into(),
            dimension: SymbolicDimension::Sequence,
        },
        TensorAxis {
            name: "width".into(),
            dimension: SymbolicDimension::Known(130),
        },
    ]);
    let mut raw = raw();
    raw.selections.truncate(1);
    raw.selections[0].transform = CaptureTransform::Preview { max_elements: 0 };
    let source = admit(raw, p, 4, false);
    let policy = CapturePrefillObservationPolicy::new(&source, g()).unwrap();
    let row = policy.row(0).unwrap();
    let shape = [2, 3, 130];
    let full = GeneratedTensorProgram::BlockFp8Input(
        BlockFp8InputReconstructionPlan::new(&shape).unwrap(),
    );
    let usage = row.full_program_usage(charge(), full).unwrap();
    assert_eq!(
        usage.retained_bytes,
        charge().retained_bytes + 4 * 6 * 256 + 12 * 6 * 130
    );
    let physical_shape = [2, 2, 130];
    assert!(row
        .full_program_usage(
            charge(),
            GeneratedTensorProgram::BlockFp8Input(
                BlockFp8InputReconstructionPlan::new(&physical_shape).unwrap()
            )
        )
        .is_err());
    let mut progress = row.initial_progress();
    let mut ledger = logical(&source);
    let before = ledger.total();
    let mut factories = 0;
    for k in 0..2 {
        let f = row.assembly().unwrap().fragment(k).unwrap();
        assert_eq!(f.output_elements(), 0);
        let decision = row
            .begin_hook(&mut progress, &chunk(&f), "block.output")
            .unwrap();
        if decision == D::First {
            row.reserve_first(&mut progress, &mut ledger, usage)
                .unwrap();
        }
        if row.factory_required(&f).unwrap() {
            factories += 1;
        }
        row.finish_hook(&mut progress, &f).unwrap();
        row.validate_chunk_end(&progress, k).unwrap();
        progress.advance_preflighted();
    }
    assert_eq!(
        factories, 2,
        "one logical creation charge, two nonempty physical factories"
    );
    assert_eq!(ledger.total(), before.checked_add(usage).unwrap());
}

#[test]
fn exact_host_envelope_prices_progress_handles_before_private_target_construction() {
    let source = rows(5);
    let h = plan(&source).initialization_peak_bytes();
    assert!(
        plan(&source).control_peak_bytes()
            >= 2 * size_of::<capture_tensor::prefill::TargetSlot>() as u64
    );
    for bytes in [h - 1, h] {
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (r, run) = fresh(&pool, bytes);
        let allocations = CLAIM_ALLOCATIONS.get();
        let result = run.prepare_capture_run(&r, plan(&source));
        if bytes < h {
            assert!(matches!(
                result,
                Err(CaptureRunHostError::Memory(
                    WorkingMemoryError::BudgetExceeded { .. }
                ))
            ));
            assert_eq!(CLAIM_ALLOCATIONS.get(), allocations);
        } else {
            let mut bank = result.unwrap();
            let step = bank
                .begin_step(CapturePhase::Prefill, 0)
                .unwrap()
                .prepare_prefill_with_progression(g())
                .unwrap();
            assert_eq!(pool.used_bytes().unwrap(), h);
            drop(step);
            drop(bank);
        }
        drop((r, run));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn schedule_inactive_geometry_needs_no_hook_and_true_empty_slice_skips_factory() {
    let mut inactive = raw();
    for selection in &mut inactive.selections {
        selection.schedule.prefill = false;
    }
    // Context axes cannot be assembled, but an inactive row does not assemble.
    let source = admit(inactive, point(), 4, false);
    let policy = CapturePrefillObservationPolicy::new(&source, g()).unwrap();
    let row = policy.row(0).unwrap();
    let mut progress = row.initial_progress();
    assert!(row.assembly().is_none());
    for k in 0..2 {
        row.validate_chunk_end(&progress, k).unwrap();
        progress.advance_preflighted();
    }
    assert!(progress.finished());
    let original = rows(5);
    let mut raw = original.admission().plan().clone();
    raw.selections.truncate(1);
    raw.selections[0].slices.push(CaptureSlice {
        axis: "row".into(),
        start: 0,
        end: 0,
        stride: 1,
    });
    let source = admit(raw, original.admission().points()[0].clone(), 4, false);
    let policy = CapturePrefillObservationPolicy::new(&source, g()).unwrap();
    let row = policy.row(0).unwrap();
    for k in 0..2 {
        let fragment = row.assembly().unwrap().fragment(k).unwrap();
        assert_eq!(fragment.output_elements(), 0);
        assert!(!row.factory_required(&fragment).unwrap());
    }
    let mut wrong = g();
    wrong.cached_positions = u64::MAX;
    assert!(CapturePrefillObservationPolicy::new(&source, wrong).is_err());
}

#[test]
fn partial_host_failure_keeps_original_buffer_and_usage_until_abort_retirement() {
    let source = rows(5);
    let policy = CapturePrefillObservationPolicy::new(&source, g()).unwrap();
    let row = policy.row(0).unwrap();
    let fragment = row.assembly().unwrap().fragment(0).unwrap();
    let h = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare_prefill_with_progression(g())
        .unwrap();
    let mut ledger = logical(&source);
    step.begin_prefill_hook(0, &chunk(&fragment), "block.output")
        .unwrap();
    step.reserve_prefill_hook(0, &mut ledger, TensorDtype::F32, charge())
        .unwrap();
    let used = ledger.total();
    {
        let mut writer = step
            .take_prefill_fragment(0, &fragment)
            .unwrap()
            .prepare()
            .unwrap();
        writer.push_f32(17.).unwrap();
    }
    let storage = step.prefill_storage(0).unwrap();
    step.fail_prefill_hook(0);
    assert!(step.complete_prefill_chunk(0).is_err());
    assert!(step
        .begin_prefill_hook(0, &chunk(&fragment), "block.output")
        .is_err());
    assert_eq!(step.prefill_storage(0).unwrap(), storage);
    assert_eq!(ledger.total(), used);
    assert_eq!(pool.used_bytes().unwrap(), h);
    drop(step);
    drop((bank, r, run));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn host_skip_retires_unstarted_targets_and_host_fail_cannot_retry() {
    for skip in [true, false] {
        let source = limited(skip);
        let policy = CapturePrefillObservationPolicy::new(&source, g()).unwrap();
        let h = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(h, 0).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
        let mut step = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare_prefill_with_progression(g())
            .unwrap();
        let mut ledger = logical(&source);
        let metadata = ledger.total();
        let row = policy.row(0).unwrap();
        let fragment = row.assembly().unwrap().fragment(0).unwrap();
        let actual = chunk(&fragment);
        if !skip {
            assert_eq!(
                step.begin_prefill_hook(0, &actual, "block.output").unwrap(),
                D::First
            );
            assert!(matches!(
                host_error(
                    step.reserve_prefill_hook(0, &mut ledger, TensorDtype::F32, charge())
                        .unwrap_err()
                ),
                E::Quota(CaptureError::Limit { .. })
            ));
            assert!(matches!(
                host_error(
                    step.begin_prefill_hook(0, &actual, "block.output")
                        .unwrap_err()
                ),
                E::Attempt
            ));
            assert!(step.complete_prefill_chunk(0).is_err());
            assert!(step.finish_prefill_targets().is_err());
        } else {
            for k in 0..2 {
                for i in 0..2 {
                    let row = policy.row(i).unwrap();
                    let fragment = row.assembly().unwrap().fragment(k).unwrap();
                    let decision = step
                        .begin_prefill_hook(i, &chunk(&fragment), "block.output")
                        .unwrap();
                    if k == 0 {
                        assert_eq!(decision, D::First);
                        assert!(matches!(
                            step.reserve_prefill_hook(i, &mut ledger, TensorDtype::F32, charge())
                                .unwrap(),
                            Some(CaptureSkipReason::Limit { .. })
                        ));
                    } else {
                        assert_eq!(decision, D::Ignore);
                    }
                    assert!(step.prefill_storage(i).is_none());
                }
                step.complete_prefill_chunk(k).unwrap();
            }
            step.finish_prefill_targets().unwrap();
            for record in step.records() {
                assert!(matches!(
                    record.outcome,
                    CaptureOutcome::Skipped {
                        reason: CaptureSkipReason::Limit { .. }
                    }
                ));
                assert!(record.payload.is_none());
                assert_eq!(record.charged.captures, 0);
            }
        }
        assert_eq!(ledger.total(), metadata);
        assert_eq!(pool.used_bytes().unwrap(), h);
        drop(step);
        drop((bank, r, run));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

//! Neutral mechanism fixtures use actual scheduled custody and actual scope
//! slots. They do not fabricate a canonical request/chunk stamp or native work.
use super::*;
use std::sync::{
    Arc, TryLockError,
    atomic::{AtomicUsize, Ordering},
};

fn segment(
    bank: &PreparedCaptureRun,
    native: &mut WorkingMemoryFundingScope,
) -> CaptureSourceSegment {
    // Crate-internal bootstrap only, using the actual bank's existing custody.
    // Production construction remains inaccessible until the canonical wrapper.
    bank.custody.begin_source_segment(native).unwrap()
}
fn fill<K: Ord + Send + 'static>(
    mut transfer: CapturePrefillFragmentTransfer<'_, '_, '_, '_, '_, K>,
) {
    transfer.validate().unwrap();
    let fragment = transfer.fragment();
    let rows = (fragment.input().end - fragment.input().start) as usize;
    for map in fragment.mappings() {
        let source = map.source_index();
        let column = source % 2;
        let row = (source / 2) % rows;
        let head = source / (2 * rows);
        transfer
            .push_f32((head * 100 + (fragment.input().start as usize + row) * 10 + column) as f32)
            .unwrap();
    }
    transfer.finish().unwrap();
}
fn retained(pool: &MemoryLedger, key: u32, bytes: u64, expected: bool) {
    assert_eq!(
        pool.registered_capacity_for_test(&key).unwrap() == Some(bytes),
        expected
    );
}

#[test]
fn private_transfer_uses_exact_original_h_and_one_target_buffer_across_fragments() {
    let source = rows(5);
    let h = plan(&source).initialization_peak_bytes();
    for bytes in [h - 1, h] {
        let pool = capture_test_ledger(h + 37, 0).unwrap();
        let actual = pool.register_host_storage([(1u32, 37), (2, 0)]).unwrap();
        let (r, run) = fresh(&pool, bytes);
        let mut native = run.scope().unwrap();
        let allocations = CLAIM_ALLOCATIONS.get();
        let result = run.prepare_capture_run(&r, plan(&source));
        if bytes < h {
            assert!(
                matches!(result,Err(CaptureRunHostError::Memory(WorkingMemoryError::DomainAllowanceExceeded{required_bytes,available_bytes, .. })) if required_bytes==h&&available_bytes==bytes)
            );
            assert_eq!(CLAIM_ALLOCATIONS.get(), allocations);
            drop(actual);
            drop(r);
            drop(run);
            native.certify().unwrap();
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);
            continue;
        }
        let mut bank = result.unwrap();
        let mut channel = segment(&bank, &mut native);
        let mut step = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare_prefill(geometry())
            .unwrap();
        step.record_skip(
            1,
            CaptureSkipReason::NotInvoked,
            None,
            CaptureUsage::default(),
        )
        .unwrap();
        step.begin_prefill_target(0, TensorDtype::F16, charge())
            .unwrap();
        let assembly =
            CapturePrefillRowAssembly::prepare(source.admission(), 0, geometry()).unwrap();
        let held = ledger(&pool);
        let mut pointer = None;
        for chunk in 0..3 {
            let fragment = assembly.fragment(chunk).unwrap();
            let claim = step.take_prefill_fragment(0, &fragment).unwrap();
            assert!(std::ptr::eq(claim.fragment(), &fragment));
            let transfer = claim
                .prepare_segment_transfer(&mut native, &mut channel, actual.clone())
                .unwrap();
            assert!(std::ptr::eq(transfer.fragment(), &fragment));
            fill(transfer);
            let storage = step.prefill_storage(0).unwrap();
            assert_eq!((storage.1, storage.2), (12, 12));
            assert_eq!(storage.3, (chunk as usize + 1) * 4);
            if let Some(pointer) = pointer {
                assert_eq!(storage.0, pointer);
            } else {
                pointer = Some(storage.0);
            }
            assert_eq!(ledger(&pool), held);
            step.complete_prefill_chunk(chunk).unwrap();
        }
        step.finish_prefill_targets().unwrap();
        assert_eq!(
            values(tensor(&step.records()[0])),
            &[
                0., 1., 10., 11., 20., 21., 100., 101., 110., 111., 120., 121.
            ]
        );
        assert_eq!(
            values(tensor(&step.records()[0])).as_ptr(),
            pointer.unwrap()
        );
        let alias = tensor(&step.records()[0]).clone();
        let usage = charged(&step);
        let frame = step
            .finish(CaptureStepOutcome::Committed, usage, usage, 0.)
            .unwrap();
        drop(actual);
        channel.validate_native_scope(&native).unwrap();
        drop(channel);
        retained(&pool, 1, 37, true);
        retained(&pool, 2, 0, true);
        drop(bank);
        drop(r);
        drop(run);
        native.certify().unwrap();
        retained(&pool, 1, 37, false);
        retained(&pool, 2, 0, false);
        drop(frame);
        assert_eq!(pool.payload_used_bytes().unwrap(), h);
        drop(alias);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn sibling_scope_foreign_source_and_other_bank_reject_before_target_allocation() {
    for failure in 0..3 {
        let source = rows(5);
        let h = plan(&source).initialization_peak_bytes();
        let pool = capture_test_ledger(2 * h + 11, 0).unwrap();
        let foreign = capture_test_ledger(11, 0).unwrap();
        let actual = pool.register_host_storage([(1u32, 11)]).unwrap();
        let unrelated = foreign.register_host_storage([(1u32, 11)]).unwrap();
        let (r, run) = fresh(&pool, 2 * h);
        let mut native = run.scope().unwrap();
        let mut sibling = run.scope().unwrap();
        let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
        let other = run.prepare_capture_run(&r, plan(&source)).unwrap();
        let mut channel = segment(if failure == 2 { &other } else { &bank }, &mut native);
        let mut step = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare_prefill(geometry())
            .unwrap();
        step.begin_prefill_target(0, TensorDtype::F32, charge())
            .unwrap();
        let assembly =
            CapturePrefillRowAssembly::prepare(source.admission(), 0, geometry()).unwrap();
        let fragment = assembly.fragment(0).unwrap();
        let claim = step.take_prefill_fragment(0, &fragment).unwrap();
        let before = ledger(&pool);
        let allocations = TRANSFER_ALLOCATIONS.get();
        let result = claim.prepare_segment_transfer(
            if failure == 0 {
                &mut sibling
            } else {
                &mut native
            },
            &mut channel,
            if failure == 1 {
                unrelated.clone()
            } else {
                actual.clone()
            },
        );
        assert!(matches!(
            &result,
            Err(CaptureRunHostError::Memory(
                WorkingMemoryError::IdentityMismatch
            ))
        ));
        drop(result); // Release every potential transfer loan before inspecting the frame.
        assert_eq!(TRANSFER_ALLOCATIONS.get(), allocations);
        assert_eq!(ledger(&pool), before);
        assert!(step.prefill_storage(0).is_none());
        assert!(step.take_prefill_fragment(0, &fragment).is_err());
        drop(step);
        drop(channel);
        drop(bank);
        drop(other);
        drop(actual);
        drop(unrelated);
        drop(r);
        drop(run);
        native.certify().unwrap();
        sibling.certify().unwrap();
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        assert_eq!(foreign.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn wrong_actual_fragment_rejects_before_source_binding_or_claim_consumption() {
    let source = rows(5);
    let foreign = rows(5);
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut native = run.scope().unwrap();
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let channel = segment(&bank, &mut native);
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare_prefill(geometry())
        .unwrap();
    step.begin_prefill_target(0, TensorDtype::F32, charge())
        .unwrap();
    let assembly = CapturePrefillRowAssembly::prepare(source.admission(), 0, geometry()).unwrap();
    let other = CapturePrefillRowAssembly::prepare(foreign.admission(), 0, geometry()).unwrap();
    let wrong = other.fragment(0).unwrap();
    let later = assembly.fragment(1).unwrap();
    let allocations = TRANSFER_ALLOCATIONS.get();
    assert!(matches!(
        step.take_prefill_fragment(0, &wrong),
        Err(CaptureRunHostError::Prefill(
            CapturePrefillHostError::Identity
        ))
    ));
    assert!(matches!(
        step.take_prefill_fragment(0, &later),
        Err(CaptureRunHostError::Prefill(CapturePrefillHostError::Order))
    ));
    assert_eq!(TRANSFER_ALLOCATIONS.get(), allocations);
    assert!(step.prefill_storage(0).is_none());
    let correct = assembly.fragment(0).unwrap();
    drop(step.take_prefill_fragment(0, &correct).unwrap());
    drop(step);
    drop(channel);
    drop(bank);
    drop(r);
    drop(run);
    native.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn constructor_rollback_restores_prior_sources_but_entered_failure_keeps_new_origin() {
    for fail_constructor in [true, false] {
        let source = rows(5);
        let h = plan(&source).initialization_peak_bytes();
        let pool = capture_test_ledger(h + 24, 0).unwrap();
        let a = pool.register_host_storage([(1u32, 11), (3, 0)]).unwrap();
        let b = pool.register_host_storage([(2u32, 13)]).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut native = run.scope().unwrap();
        let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
        let mut channel = segment(&bank, &mut native);
        let mut step = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare_prefill(geometry())
            .unwrap();
        step.record_skip(
            1,
            CaptureSkipReason::NotInvoked,
            None,
            CaptureUsage::default(),
        )
        .unwrap();
        step.begin_prefill_target(0, TensorDtype::F32, charge())
            .unwrap();
        let assembly =
            CapturePrefillRowAssembly::prepare(source.admission(), 0, geometry()).unwrap();
        let first = assembly.fragment(0).unwrap();
        fill(
            step.take_prefill_fragment(0, &first)
                .unwrap()
                .prepare_segment_transfer(&mut native, &mut channel, a)
                .unwrap(),
        );
        let before = step.prefill_storage(0).unwrap();
        step.complete_prefill_chunk(0).unwrap();
        let next = assembly.fragment(1).unwrap();
        if fail_constructor {
            PANIC_TRANSFER.set(true);
        }
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let mut transfer = step
                    .take_prefill_fragment(0, &next)
                    .unwrap()
                    .prepare_segment_transfer(&mut native, &mut channel, b)
                    .unwrap();
                transfer.push_f32(55.).unwrap();
                panic!("after constructor and scalar");
            }))
            .is_err()
        );
        let partial = step.prefill_storage(0).unwrap();
        assert_eq!(partial.0, before.0);
        assert_eq!(partial.3, if fail_constructor { 4 } else { 5 });
        assert!(step.take_prefill_fragment(0, &next).is_err());
        let usage = charged(&step);
        let pending = step.into_aborted_pending(usage, usage, 0.);
        assert_eq!(pending.prefill_storage(0), Some(partial));
        drop(channel);
        drop(bank);
        drop(r);
        drop(run);
        let expected = h + if fail_constructor { 11 } else { 24 };
        assert_eq!(pool.payload_used_bytes().unwrap(), expected);
        drop(native);
        assert_eq!(pool.payload_used_bytes().unwrap(), expected);
        retained(&pool, 1, 11, true);
        retained(&pool, 3, 0, true);
        retained(&pool, 2, 13, !fail_constructor);
        drop(pending);
        assert_eq!(pool.payload_used_bytes().unwrap(), expected);
        assert!(pool.acquire_unquoted().is_err());
    }
}

#[test]
fn all_prior_origin_health_is_rechecked_during_fill_and_before_later_construction() {
    for quarantine_during_fill in [false, true] {
        let source = rows(5);
        let h = plan(&source).initialization_peak_bytes();
        let pool = capture_test_ledger(h + 19, 0).unwrap();
        let (origin_r, origin) = fresh(&pool, 19);
        let origin_native = origin.scope().unwrap();
        let owned = origin_native
            .adopt_capture_host_storage([(1u32, 19), (2, 0)])
            .unwrap();
        let (r, run) = fresh(&pool, h);
        let mut native = run.scope().unwrap();
        let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
        let mut channel = segment(&bank, &mut native);
        let mut step = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare_prefill(geometry())
            .unwrap();
        step.record_skip(
            1,
            CaptureSkipReason::NotInvoked,
            None,
            CaptureUsage::default(),
        )
        .unwrap();
        step.begin_prefill_target(0, TensorDtype::F32, charge())
            .unwrap();
        let assembly =
            CapturePrefillRowAssembly::prepare(source.admission(), 0, geometry()).unwrap();
        let first = assembly.fragment(0).unwrap();
        let pins = pool.pin_registered_storage([(1u32, 19), (2, 0)]).unwrap();
        fill(
            step.take_prefill_fragment(0, &first)
                .unwrap()
                .prepare_segment_transfer(&mut native, &mut channel, pins)
                .unwrap(),
        );
        step.complete_prefill_chunk(0).unwrap();
        let prior = step.prefill_storage(0).unwrap();
        let next = assembly.fragment(1).unwrap();
        let empty = pool
            .pin_registered_storage(Vec::<(u32, u64)>::new())
            .unwrap();
        if quarantine_during_fill {
            let mut transfer = step
                .take_prefill_fragment(0, &next)
                .unwrap()
                .prepare_segment_transfer(&mut native, &mut channel, empty)
                .unwrap();
            drop(origin_native);
            assert!(matches!(
                transfer.validate(),
                Err(CaptureRunHostError::Memory(
                    WorkingMemoryError::ExecutionFenced
                ))
            ));
            assert!(matches!(
                transfer.push_f32(77.),
                Err(CaptureRunHostError::Memory(
                    WorkingMemoryError::ExecutionFenced
                ))
            ));
            assert!(transfer.finish().is_err());
        } else {
            drop(origin_native);
            let allocations = TRANSFER_ALLOCATIONS.get();
            let result = step
                .take_prefill_fragment(0, &next)
                .unwrap()
                .prepare_segment_transfer(&mut native, &mut channel, empty);
            assert!(matches!(
                result,
                Err(CaptureRunHostError::Memory(
                    WorkingMemoryError::ExecutionFenced
                ))
            ));
            assert_eq!(TRANSFER_ALLOCATIONS.get(), allocations);
        }
        assert_eq!(step.prefill_storage(0), Some(prior));
        let usage = charged(&step);
        let pending = step.into_aborted_pending(usage, usage, 0.);
        drop(channel);
        drop(bank);
        drop(owned);
        drop(r);
        drop(run);
        drop(origin_r);
        drop(origin);
        drop(native);
        drop(pending);
        retained(&pool, 1, 19, true);
        retained(&pool, 2, 0, true);
        assert!(pool.acquire_unquoted().is_err());
    }
}

#[test]
fn short_overlong_and_closed_parent_failures_are_terminal_and_preserve_partial_h() {
    for failure in 0..3 {
        let source = rows(5);
        let h = plan(&source).initialization_peak_bytes();
        let pool = capture_test_ledger(h + 7, 0).unwrap();
        let actual = pool.register_host_storage([(1u32, 7)]).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut run = Some(run);
        let mut native = run.as_ref().unwrap().scope().unwrap();
        let mut bank = run
            .as_ref()
            .unwrap()
            .prepare_capture_run(&r, plan(&source))
            .unwrap();
        let mut channel = segment(&bank, &mut native);
        let mut step = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare_prefill(geometry())
            .unwrap();
        step.begin_prefill_target(0, TensorDtype::F32, charge())
            .unwrap();
        let assembly =
            CapturePrefillRowAssembly::prepare(source.admission(), 0, geometry()).unwrap();
        let fragment = assembly.fragment(0).unwrap();
        let mut transfer = step
            .take_prefill_fragment(0, &fragment)
            .unwrap()
            .prepare_segment_transfer(&mut native, &mut channel, actual)
            .unwrap();
        transfer.push_f32(8.).unwrap();
        if failure == 1 {
            for _ in 1..fragment.output_elements() {
                transfer.push_f32(9.).unwrap();
            }
            assert!(matches!(
                transfer.push_f32(10.),
                Err(CaptureRunHostError::Prefill(CapturePrefillHostError::Order))
            ));
            assert!(matches!(
                transfer.validate(),
                Err(CaptureRunHostError::Prefill(
                    CapturePrefillHostError::Incomplete { index: 0 }
                ))
            ));
        }
        if failure == 2 {
            run.take().unwrap().close().unwrap();
            assert!(matches!(
                transfer.push_f32(10.),
                Err(CaptureRunHostError::Memory(
                    WorkingMemoryError::ExecutionFenced
                ))
            ));
        }
        let error = transfer.finish().unwrap_err();
        assert!(matches!(
            error,
            CaptureRunHostError::Memory(WorkingMemoryError::ExecutionFenced)
                | CaptureRunHostError::Prefill(CapturePrefillHostError::Incomplete { index: 0 })
        ));
        let partial = step.prefill_storage(0).unwrap();
        assert_eq!(partial.3, if failure == 1 { 4 } else { 1 });
        assert!(step.take_prefill_fragment(0, &fragment).is_err());
        let usage = charged(&step);
        let pending = step.into_aborted_pending(usage, usage, 0.);
        assert_eq!(pending.prefill_storage(0), Some(partial));
        drop(channel);
        drop(bank);
        drop(run);
        drop(r);
        retained(&pool, 1, 7, true);
        native.certify().unwrap();
        retained(&pool, 1, 7, false);
        assert_eq!(pool.payload_used_bytes().unwrap(), h);
        drop(pending);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[derive(Clone)]
struct Key {
    id: u32,
    probe: Arc<Probe>,
}
struct Probe {
    pool: MemoryLedger,
    drops: AtomicUsize,
}
impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}
impl Drop for Key {
    fn drop(&mut self) {
        assert!(!matches!(
            self.probe.pool.0.usage.try_lock(),
            Err(TryLockError::WouldBlock)
        ));
        self.probe.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn source_key_retirement_stays_outside_usage_while_partial_target_keeps_h() {
    let source = rows(5);
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h + 7, 0).unwrap();
    let probe = Arc::new(Probe {
        pool: pool.clone(),
        drops: AtomicUsize::new(0),
    });
    let actual = pool
        .register_host_storage([(
            Key {
                id: 1,
                probe: probe.clone(),
            },
            7,
        )])
        .unwrap();
    let (r, run) = fresh(&pool, h);
    let mut native = run.scope().unwrap();
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut channel = segment(&bank, &mut native);
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare_prefill(geometry())
        .unwrap();
    step.begin_prefill_target(0, TensorDtype::F32, charge())
        .unwrap();
    let assembly = CapturePrefillRowAssembly::prepare(source.admission(), 0, geometry()).unwrap();
    let fragment = assembly.fragment(0).unwrap();
    let mut transfer = step
        .take_prefill_fragment(0, &fragment)
        .unwrap()
        .prepare_segment_transfer(&mut native, &mut channel, actual)
        .unwrap();
    transfer.push_f32(6.25).unwrap();
    drop(transfer);
    let partial = step.prefill_storage(0).unwrap();
    let usage = charged(&step);
    let pending = step.into_aborted_pending(usage, usage, 0.);
    drop(channel);
    drop(bank);
    drop(r);
    drop(run);
    let before = probe.drops.load(Ordering::SeqCst);
    native.certify().unwrap();
    assert!(probe.drops.load(Ordering::SeqCst) > before);
    assert_eq!(pending.prefill_storage(0), Some(partial));
    assert_eq!(pool.payload_used_bytes().unwrap(), h);
    drop(pending);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn empty_fragment_still_validates_and_retains_zero_byte_source_origin() {
    let source = rows(0);
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h, 0).unwrap();
    let zero = pool.register_host_storage([(1u32, 0)]).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut native = run.scope().unwrap();
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut channel = segment(&bank, &mut native);
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare_prefill(geometry())
        .unwrap();
    step.record_skip(
        0,
        CaptureSkipReason::NotInvoked,
        None,
        CaptureUsage::default(),
    )
    .unwrap();
    step.begin_prefill_target(1, TensorDtype::F16, charge())
        .unwrap();
    let assembly = CapturePrefillRowAssembly::prepare(source.admission(), 1, geometry()).unwrap();
    let fragment = assembly.fragment(0).unwrap();
    assert_eq!(fragment.output_elements(), 0);
    let held = ledger(&pool);
    let transfer = step
        .take_prefill_fragment(1, &fragment)
        .unwrap()
        .prepare_segment_transfer(&mut native, &mut channel, zero)
        .unwrap();
    transfer.validate().unwrap();
    transfer.finish().unwrap();
    assert_eq!(step.prefill_storage(1).unwrap().2, 0);
    assert_eq!(ledger(&pool), held);
    for chunk in 0..3 {
        step.complete_prefill_chunk(chunk).unwrap();
    }
    step.finish_prefill_targets().unwrap();
    let usage = charged(&step);
    let frame = step
        .finish(CaptureStepOutcome::Committed, usage, usage, 0.)
        .unwrap();
    assert!(values(tensor(&frame.as_ref().records[1])).is_empty());
    drop(channel);
    retained(&pool, 1, 0, true);
    drop(bank);
    drop(r);
    drop(run);
    native.certify().unwrap();
    retained(&pool, 1, 0, false);
    assert_eq!(pool.payload_used_bytes().unwrap(), h);
    drop(frame);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

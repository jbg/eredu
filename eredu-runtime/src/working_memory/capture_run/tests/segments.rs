use super::*;
use std::sync::{
    Arc, TryLockError,
    atomic::{AtomicUsize, Ordering},
};

fn selected_source() -> SharedCapturePlan {
    let mut raw = raw();
    raw.selections = (0..5)
        .map(|index| {
            let mut selection = raw.selections[0].clone();
            selection.id = format!("segment-{index}");
            selection
        })
        .collect();
    admit(raw, point(), 4, false)
}
fn retained(pool: &MemoryLedger, key: u32, bytes: u64, expected: bool) {
    let retained = pool.registered_capacity_for_test(&key).unwrap();
    assert_eq!(retained, expected.then_some(bytes));
}
fn fill<'a, 'c, 's, K: Ord + Send + 'static>(
    mut transfer: ScheduledCaptureTensorTransfer<'a, 'c, 's, K>,
) -> ClaimedCaptureTensor {
    while transfer.initialized_count() < transfer.len() {
        transfer.push_f32(2.75).unwrap();
    }
    transfer.finish().unwrap()
}

#[test]
fn segment_exact_original_h_has_no_second_hold_and_short_h_rejects_before_construction() {
    let source = selected_source();
    let h = plan(&source).initialization_peak_bytes();
    for bytes in [h - 1, h] {
        let pool = capture_test_ledger(bytes, 0).unwrap();
        let (r, run) = fresh(&pool, bytes);
        let mut native = run.scope().unwrap();
        let before = ledger(&pool);
        let allocations = CLAIM_ALLOCATIONS.get();
        if bytes < h {
            assert!(matches!(
                run.prepare_capture_run(&r, plan(&source)),
                Err(CaptureRunHostError::Memory(
                    WorkingMemoryError::DomainAllowanceExceeded { .. }
                ))
            ));
            assert_eq!(CLAIM_ALLOCATIONS.get(), allocations);
            assert_eq!(ledger(&pool), before);
        } else {
            let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
            let mut step = bank
                .begin_step(CapturePhase::Prefill, 0)
                .unwrap()
                .prepare()
                .unwrap();
            let claim = step.take_tensor(0).unwrap();
            let held = ledger(&pool);
            let mut segment = claim.begin_source_segment(&mut native).unwrap();
            let empty = pool
                .pin_registered_storage(Vec::<(u32, u64)>::new())
                .unwrap();
            let receipt = fill(
                claim
                    .prepare_with_segment_source(&mut native, &mut segment, empty)
                    .unwrap(),
            );
            assert_eq!(ledger(&pool), held);
            step.record_tensor(receipt, TensorDtype::F32, CaptureUsage::default())
                .unwrap();
            drop(finish(step));
            segment.retire_after_settled_boundary(&mut native).unwrap();
            assert_eq!(ledger(&pool), held);
            drop((segment, bank));
        }
        drop((r, run));
        native.certify().unwrap();
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn segment_retirement_preserves_baseline_interleaved_pins_aliases_and_zero_keys() {
    let source = selected_source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h + 43, 0).unwrap();
    let a = pool.register_host_storage([(1u32, 11)]).unwrap();
    let b = pool.register_host_storage([(2u32, 13)]).unwrap();
    let escaped_b = b.clone();
    let c = pool.register_host_storage([(3u32, 19)]).unwrap();
    let zero = pool.register_host_storage([(4u32, 0)]).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut native = run.scope().unwrap();
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    drop(fill(
        step.take_tensor(0)
            .unwrap()
            .prepare_with_source(&mut native, a.clone())
            .unwrap(),
    ));
    let claim = step.take_tensor(1).unwrap();
    let mut segment = claim.begin_source_segment(&mut native).unwrap();
    drop(fill(
        claim
            .prepare_with_segment_source(&mut native, &mut segment, b)
            .unwrap(),
    ));
    // Permanent additions made after channel creation must survive too.
    drop(fill(
        step.take_tensor(2)
            .unwrap()
            .prepare_with_source(&mut native, c)
            .unwrap(),
    ));
    drop(fill(
        step.take_tensor(3)
            .unwrap()
            .prepare_with_segment_source(&mut native, &mut segment, zero)
            .unwrap(),
    ));
    drop(fill(
        step.take_tensor(4)
            .unwrap()
            .prepare_with_segment_source(&mut native, &mut segment, a)
            .unwrap(),
    ));
    drop(finish(step));
    assert_eq!(pool.payload_used_bytes().unwrap(), h + 43);
    let held = ledger(&pool).1;
    segment.retire_after_settled_boundary(&mut native).unwrap();
    assert!(matches!(
        segment.validate_native_scope(&native),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        segment.retire_after_settled_boundary(&mut native),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(ledger(&pool).1, held);
    retained(&pool, 1, 11, true);
    retained(&pool, 2, 13, true); // External alias keeps this actual charge alive.
    retained(&pool, 3, 19, true);
    retained(&pool, 4, 0, false);
    drop(escaped_b);
    assert_eq!(pool.payload_used_bytes().unwrap(), h + 30);
    drop((segment, bank, r, run));
    native.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn segment_rejects_sibling_scope_independent_schedule_and_foreign_source_before_allocation() {
    let source = selected_source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(2 * h + 11, 0).unwrap();
    let foreign = capture_test_ledger(11, 0).unwrap();
    let storage = pool.register_host_storage([(1u32, 11)]).unwrap();
    let foreign_storage = foreign.register_host_storage([(1u32, 11)]).unwrap();
    let (r, run) = fresh(&pool, 2 * h);
    let mut native = run.scope().unwrap();
    let mut sibling = run.scope().unwrap();
    let mut a = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut b = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = a
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let mut other = b
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let claim = step.take_tensor(0).unwrap();
    let mut segment = claim.begin_source_segment(&mut native).unwrap();
    let sibling_segment = claim.begin_source_segment(&mut sibling).unwrap();
    assert!(claim.begin_source_segment(&mut native).is_err());
    let before = ledger(&pool);
    let allocations = TRANSFER_ALLOCATIONS.get();
    assert!(matches!(
        segment.validate_native_scope(&sibling),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(
        claim
            .prepare_with_segment_source(&mut sibling, &mut segment, storage.clone())
            .is_err()
    );
    assert!(
        step.take_tensor(1)
            .unwrap()
            .prepare_with_segment_source(&mut native, &mut segment, foreign_storage)
            .is_err()
    );
    assert!(
        other
            .take_tensor(0)
            .unwrap()
            .prepare_with_segment_source(&mut native, &mut segment, storage.clone())
            .is_err()
    );
    assert_eq!(TRANSFER_ALLOCATIONS.get(), allocations);
    assert_eq!(ledger(&pool), before);
    drop((step, other));
    segment.retire_after_settled_boundary(&mut native).unwrap();
    drop((segment, sibling_segment, a, b, storage, r, run));
    native.certify().unwrap();
    sibling.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert_eq!(foreign.payload_used_bytes().unwrap(), 0);
}

#[test]
fn segment_constructor_rollback_and_abandonment_preserve_both_quarantine_channels() {
    for fail_constructor in [true, false] {
        let source = selected_source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = capture_test_ledger(h + 43, 0).unwrap();
        let a = pool.register_host_storage([(1u32, 11)]).unwrap();
        let b = pool.register_host_storage([(2u32, 13)]).unwrap();
        let c = pool.register_host_storage([(3u32, 19)]).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut native = run.scope().unwrap();
        let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
        let mut step = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap();
        drop(fill(
            step.take_tensor(0)
                .unwrap()
                .prepare_with_source(&mut native, a)
                .unwrap(),
        ));
        let claim = step.take_tensor(1).unwrap();
        let mut segment = claim.begin_source_segment(&mut native).unwrap();
        drop(fill(
            claim
                .prepare_with_segment_source(&mut native, &mut segment, b)
                .unwrap(),
        ));
        if fail_constructor {
            PANIC_TRANSFER.set(true);
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    drop(step.take_tensor(2).unwrap().prepare_with_segment_source(
                        &mut native,
                        &mut segment,
                        c,
                    ));
                }))
                .is_err()
            );
        } else {
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    let mut partial = step
                        .take_tensor(2)
                        .unwrap()
                        .prepare_with_segment_source(&mut native, &mut segment, c)
                        .unwrap();
                    partial.push_f32(7.).unwrap();
                    panic!("entered worker after initialized scalar");
                }))
                .is_err()
            ); // Entered worker: retain this origin despite unwind.
        }
        assert!(step.take_tensor(2).is_err());
        drop(step);
        drop(segment); // Handle Drop is not successful segment retirement.
        drop((bank, r, run));
        let expected = h + if fail_constructor { 24 } else { 43 };
        assert_eq!(pool.payload_used_bytes().unwrap(), expected);
        drop(native);
        assert_eq!(pool.payload_used_bytes().unwrap(), expected);
        retained(&pool, 1, 11, true);
        retained(&pool, 2, 13, true);
        retained(&pool, 3, 19, !fail_constructor);
        assert!(pool.acquire_unquoted().is_err());
    }
}

#[test]
fn segment_checks_all_prior_origins_and_parent_health_before_more_work_or_retirement() {
    for close_parent in [false, true] {
        let source = selected_source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = capture_test_ledger(h + 19, 0).unwrap();
        let (origin_r, origin) = fresh(&pool, 19);
        let origin_native = origin.scope().unwrap();
        let a = origin_native
            .adopt_capture_host_storage([(1u32, 19), (2, 0)])
            .unwrap();
        let (r, run) = fresh(&pool, h);
        let mut run = Some(run);
        let mut native = run.as_ref().unwrap().scope().unwrap();
        let mut bank = run
            .as_ref()
            .unwrap()
            .prepare_capture_run(&r, plan(&source))
            .unwrap();
        let mut step = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap();
        let claim = step.take_tensor(0).unwrap();
        let mut segment = claim.begin_source_segment(&mut native).unwrap();
        let pins = pool.pin_registered_storage([(1u32, 19), (2, 0)]).unwrap();
        drop(fill(
            claim
                .prepare_with_segment_source(&mut native, &mut segment, pins)
                .unwrap(),
        ));
        let next = step.take_tensor(1).unwrap();
        if close_parent {
            run.take().unwrap().close().unwrap(); // Origin remains healthy: test run closure independently.
            origin_native.certify().unwrap();
        } else {
            drop(origin_native); // Parent still open: test origin quarantine independently.
        }
        let allocations = TRANSFER_ALLOCATIONS.get();
        assert!(matches!(
            segment.validate_native_scope(&native),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        assert!(matches!(
            segment.retire_after_settled_boundary(&mut native),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        let empty = pool
            .pin_registered_storage(Vec::<(u32, u64)>::new())
            .unwrap();
        assert!(
            next.prepare_with_segment_source(&mut native, &mut segment, empty)
                .is_err()
        );
        assert_eq!(TRANSFER_ALLOCATIONS.get(), allocations);
        drop(step);
        drop((segment, bank, a, r, run, origin_r, origin, native));
        retained(&pool, 1, 19, true);
        retained(&pool, 2, 0, true);
        assert!(pool.acquire_unquoted().is_err());
    }
}

#[test]
fn segment_release_returns_only_actual_retired_allocation_credit_to_original_account() {
    let source = selected_source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h + 37, 0).unwrap();
    let (r, run) = fresh(&pool, h + 37);
    let mut native = run.scope().unwrap();
    let owner = native
        .adopt_capture_host_storage([(1u32, 37)])
        .unwrap()
        .into_values()
        .next()
        .unwrap();
    let escaped_owner = owner.clone();
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let claim = step.take_tensor(0).unwrap();
    let mut segment = claim.begin_source_segment(&mut native).unwrap();
    let receipt = fill(
        claim
            .prepare_with_segment_source(&mut native, &mut segment, owner)
            .unwrap(),
    );
    step.record_tensor(receipt, TensorDtype::F32, CaptureUsage::default())
        .unwrap();
    let frame = finish(step);
    segment.retire_after_settled_boundary(&mut native).unwrap();
    {
        let usage = pool.0.usage.lock().unwrap();
        assert_eq!(usage.registered - usage.registry_metadata, 37);
    }
    drop(escaped_owner);
    let usage = pool.0.usage.lock().unwrap();
    assert_eq!(usage.registered, 0);
    assert_eq!(
        usage.reserved - usage.funding.control_bytes().unwrap(),
        h + 37
    );
    drop(usage);
    assert_eq!(ledger(&pool).1, h);
    drop((segment, bank, r, run));
    native.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), h); // Only the frame's full original H survives.
    drop(frame);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[derive(Clone)]
struct ProbeKey {
    id: u32,
    probe: Arc<Probe>,
}
struct Probe {
    pool: MemoryLedger,
    drops: AtomicUsize,
}
impl PartialEq for ProbeKey {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for ProbeKey {}
impl PartialOrd for ProbeKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ProbeKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}
impl Drop for ProbeKey {
    fn drop(&mut self) {
        assert!(!matches!(
            self.probe.pool.0.usage.try_lock(),
            Err(TryLockError::WouldBlock)
        ));
        self.probe.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn segment_source_key_retirement_runs_outside_usage_and_keeps_original_h() {
    let source = selected_source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h + 7, 0).unwrap();
    let probe = Arc::new(Probe {
        pool: pool.clone(),
        drops: AtomicUsize::new(0),
    });
    let storage = pool
        .register_host_storage([(
            ProbeKey {
                id: 1,
                probe: probe.clone(),
            },
            7,
        )])
        .unwrap();
    let (r, run) = fresh(&pool, h);
    let mut native = run.scope().unwrap();
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let claim = step.take_tensor(0).unwrap();
    let mut segment = claim.begin_source_segment(&mut native).unwrap();
    drop(fill(
        claim
            .prepare_with_segment_source(&mut native, &mut segment, storage)
            .unwrap(),
    ));
    drop(finish(step));
    let before = probe.drops.load(Ordering::SeqCst);
    segment.retire_after_settled_boundary(&mut native).unwrap();
    assert!(probe.drops.load(Ordering::SeqCst) > before);
    assert_eq!(pool.payload_used_bytes().unwrap(), h);
    assert_eq!(ledger(&pool).1, h);
    drop((segment, bank, r, run));
    native.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

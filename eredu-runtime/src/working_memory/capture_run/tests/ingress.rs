use super::*;

#[test]
fn pre_evaluation_scope_check_rejects_foreign_accounts_without_storage_or_claim_refund() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h * 2, 0).unwrap();
    let foreign = capture_test_ledger(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let (other_r, other_run) = fresh(&pool, h);
    let (foreign_r, foreign_run) = fresh(&foreign, h);
    let native = run.scope().unwrap();
    let other = other_run.scope().unwrap();
    let foreign_native = foreign_run.scope().unwrap();
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let claim = step.take_tensor(0).unwrap();
    let before = ledger(&pool);
    let allocations = TRANSFER_ALLOCATIONS.get();
    claim.validate_native_scope(&native).unwrap();
    for scope in [&other, &foreign_native] {
        assert_eq!(
            claim.validate_native_scope(scope),
            Err(WorkingMemoryError::IdentityMismatch)
        );
    }
    assert_eq!(ledger(&pool), before);
    assert_eq!(TRANSFER_ALLOCATIONS.get(), allocations);
    drop(claim);
    assert!(matches!(
        step.take_tensor(0),
        Err(CaptureRunHostError::ClaimUnavailable { index: 0 })
    ));
    drop(step);
    drop(bank);
    native.certify().unwrap();
    other.certify().unwrap();
    foreign_native.certify().unwrap();
    drop((r, run, other_r, other_run, foreign_r, foreign_run));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert_eq!(foreign.payload_used_bytes().unwrap(), 0);
}

#[test]
fn pre_evaluation_scope_check_rejects_closed_and_quarantined_parent() {
    for quarantine in [false, true] {
        let source = source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = capture_test_ledger(h, 0).unwrap();
        let (r, run) = fresh(&pool, h);
        let native = run.scope().unwrap();
        let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
        let mut step = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap();
        let claim = step.take_tensor(0).unwrap();
        claim.validate_native_scope(&native).unwrap();
        if quarantine {
            drop(run.scope().unwrap());
        } else {
            drop(run);
        }
        let before = ledger(&pool);
        assert_eq!(
            claim.validate_native_scope(&native),
            Err(WorkingMemoryError::ExecutionFenced)
        );
        assert_eq!(ledger(&pool), before);
        drop(claim);
        drop(step);
        drop(bank);
        if !quarantine {
            native.certify().unwrap();
            drop(r);
            assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        } else {
            drop(native);
            drop(r);
            assert_eq!(pool.payload_used_bytes().unwrap(), h);
        }
    }
}

#[test]
fn terminal_scalar_error_releases_borrow_but_retains_whole_original_hold() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h, 0).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let mut partial = step.take_tensor(0).unwrap().prepare().unwrap();
    let expected = partial.len();
    partial.push_f32(-3.25).unwrap();
    let failure = partial.finish().unwrap_err().into_owned_error();
    assert_eq!(
        failure.to_string(),
        format!("capture tensor has 1 of {expected} values")
    );
    assert!(std::error::Error::source(&failure).is_none());
    assert!(step.take_tensor(0).is_err());
    drop(step);
    drop((bank, r, run, source));
    // The terminal owner is 'static and carries H after the source/frame died.
    let erased: Box<dyn std::error::Error + Send + Sync> = Box::new(failure);
    assert_eq!(pool.payload_used_bytes().unwrap(), h);
    drop(erased);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn terminal_transfer_error_releases_exact_scope_borrow_without_certification() {
    let source = source();
    let h = plan(&source).initialization_peak_bytes();
    let pool = capture_test_ledger(h + 19, 0).unwrap();
    let storage = pool.register_host_storage([(1u32, 19)]).unwrap();
    let (r, run) = fresh(&pool, h);
    let mut native = run.scope().unwrap();
    let sibling = run.scope().unwrap();
    let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
    let mut step = bank
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap()
        .prepare()
        .unwrap();
    let mut partial = step
        .take_tensor(0)
        .unwrap()
        .prepare_with_source(&mut native, storage)
        .unwrap();
    partial.push_f32(4.5).unwrap();
    sibling.certify().unwrap();
    let failure = partial.finish().unwrap_err().into_owned_error();
    assert!(step.take_tensor(0).is_err());
    drop(step);
    drop((bank, r, run, source));
    assert_eq!(pool.payload_used_bytes().unwrap(), h + 19);
    native.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), h);
    drop(failure);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn terminal_rejections_move_original_typed_cause_and_never_resume_or_refund() {
    for transfer in [false, true] {
        let source = source();
        let h = plan(&source).initialization_peak_bytes();
        let pool = capture_test_ledger(h, 0).unwrap();
        let storage = pool.register_storage::<u32>([]).unwrap();
        let (r, run) = fresh(&pool, h);
        let mut native = run.scope().unwrap();
        let mut bank = run.prepare_capture_run(&r, plan(&source)).unwrap();
        let mut step = bank
            .begin_step(CapturePhase::Prefill, 0)
            .unwrap()
            .prepare()
            .unwrap();
        let claim = step.take_tensor(0).unwrap();
        let failure = if transfer {
            let mut partial = claim.prepare_with_source(&mut native, storage).unwrap();
            while partial.initialized_count() < partial.len() {
                partial.push_f32(1.25).unwrap();
            }
            drop(run);
            partial.finish().unwrap_err().into_owned_error()
        } else {
            let mut partial = claim.prepare().unwrap();
            while partial.initialized_count() < partial.len() {
                partial.push_f32(-2.0).unwrap();
            }
            drop(run);
            partial.finish().unwrap_err().into_owned_error()
        };
        assert_eq!(
            std::error::Error::source(&failure)
                .unwrap()
                .downcast_ref::<WorkingMemoryError>(),
            Some(&WorkingMemoryError::ExecutionFenced)
        );
        drop(step);
        drop((bank, r));
        native.certify().unwrap();
        assert_eq!(pool.payload_used_bytes().unwrap(), h);
        drop(failure);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

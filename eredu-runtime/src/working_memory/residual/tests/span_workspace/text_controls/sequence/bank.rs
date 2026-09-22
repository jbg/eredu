use super::*;
use eredu_core::{BackendFailureKind, GenerationSequenceBankRejection as Rejection};
use std::sync::Arc;

fn memory_cause(error: &eredu_core::BackendFailure) -> &WorkingMemoryError {
    error
        .source()
        .unwrap()
        .downcast_ref::<RuntimeSequenceError>()
        .unwrap()
        .cause()
        .source()
        .unwrap()
        .downcast_ref::<WorkingMemoryError>()
        .unwrap()
}

#[test]
fn original_bank_many_retained_rejections_do_not_retain_or_clone_custody() {
    let (mut runtime, state) = runtime(Mode::default());
    let sequence = extract(&mut runtime, 3, &[], false, false, None).unwrap();
    drop(sequence);
    let preparation = Rc::clone(state.borrow().active.as_ref().unwrap());
    let before = Arc::strong_count(preparation.owner.borrow().reservation().0.inner());
    let errors = std::array::from_fn::<_, 256, _>(|i| {
        assert!(
            preparation
                .owner
                .borrow_mut()
                .take_generation_sequence_bank()
                .is_none()
        );
        match i % 3 {
            0 => Rejection::Unavailable,
            1 => Rejection::Busy,
            _ => Rejection::IdentityMismatch,
        }
        .into_backend_failure()
    });
    assert_eq!(
        Arc::strong_count(preparation.owner.borrow().reservation().0.inner()),
        before
    );
    drop(preparation);
    let (pool, _) = retire_request(&state);
    drop(runtime);
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        0,
        "all rejection values still live"
    );
    for error in &errors {
        assert!(error.source().unwrap().is::<Rejection>());
    }
    drop(errors);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn original_bank_foreign_genuine_claim_spends_one_bank_and_escapes_with_its_owner() {
    let (mut first, old) = runtime(Mode {
        defer_bank: true,
        explicit_source: true,
        ..Mode::default()
    });
    let initial = extract(&mut first, 3, &[7], false, false, None).unwrap_err();
    assert!(initial.source().unwrap().is::<Rejection>());
    let original = Rc::clone(old.borrow().active.as_ref().unwrap());
    let bank = old.borrow_mut().pending_bank.take().unwrap();
    assert!(
        original
            .owner
            .borrow_mut()
            .take_generation_sequence_bank()
            .is_none()
    );
    let (mut second, new) = runtime(Mode::default());
    new.borrow_mut().foreign = Some(Rc::clone(&original));
    new.borrow_mut().pending_bank = Some(bank);
    let error = extract(&mut second, 3, &[7], false, false, None).unwrap_err();
    assert!(matches!(
        memory_cause(&error),
        WorkingMemoryError::IdentityMismatch
    ));
    assert_eq!(error.kind(), BackendFailureKind::InvalidSession);
    assert_eq!(new.borrow().order, vec!["admit", "bind", "extract"]);
    assert_eq!(new.borrow().votes, vec![(Stage::Admission, Status::Failed)]);
    let errors = std::array::from_fn::<_, 128, _>(|_| {
        assert!(
            original
                .owner
                .borrow_mut()
                .take_generation_sequence_bank()
                .is_none()
        );
        Rejection::Unavailable.into_backend_failure()
    });
    drop(original);
    let (old_pool, old_held) = retire_request(&old);
    drop(first);
    let (new_pool, _) = retire_request(&new);
    drop(second);
    assert_eq!(new_pool.payload_used_bytes().unwrap(), 0);
    assert_eq!(old_pool.payload_used_bytes().unwrap(), old_held + 64);
    drop(diagnostic_pin(&old, &old_pool).unwrap());
    drop(error);
    assert!(matches!(
        diagnostic_pin(&old, &old_pool),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(
        old_pool.payload_used_bytes().unwrap(),
        0,
        "replays and initial static rejection still live"
    );
    drop((errors, initial));
}

#[test]
fn original_bank_fenced_consuming_error_owns_original_reservation_before_prompt() {
    let (mut runtime, state) = runtime(Mode {
        fence_bank: true,
        ..Mode::default()
    });
    let error = extract(&mut runtime, 4, &[9], true, false, None).unwrap_err();
    assert!(matches!(
        memory_cause(&error),
        WorkingMemoryError::ExecutionFenced
    ));
    assert_eq!(state.borrow().order, vec!["admit", "bind", "extract"]);
    assert!(
        state
            .borrow()
            .active
            .as_ref()
            .unwrap()
            .owner
            .borrow_mut()
            .take_generation_sequence_bank()
            .is_none()
    );
    let (pool, held) = retire_request(&state);
    drop(runtime);
    assert!(matches!(
        diagnostic_pin(&state, &pool),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), held);
    drop(error);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn original_bank_unwind_after_take_cannot_restore_the_bank() {
    let (mut runtime, state) = runtime(Mode {
        panic_after_take: true,
        ..Mode::default()
    });
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = extract(&mut runtime, 3, &[], false, false, None);
    }))
    .unwrap_err();
    assert_eq!(
        panic.downcast_ref::<&str>(),
        Some(&"consumed original sequence bank unwind")
    );
    assert!(
        state
            .borrow()
            .active
            .as_ref()
            .unwrap()
            .owner
            .borrow_mut()
            .take_generation_sequence_bank()
            .is_none()
    );
    assert_eq!(state.borrow().order, vec!["admit", "bind", "extract"]);
    let (pool, _) = retire_request(&state);
    drop(runtime);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn original_bank_consuming_storage_error_retires_before_each_recovery() {
    let (mut runtime, state) = runtime(Mode::default());
    let sequence = extract(&mut runtime, 3, &[], false, false, None).unwrap();
    state
        .borrow()
        .active
        .as_ref()
        .unwrap()
        .run
        .borrow_mut()
        .take();
    let error = sequence.prepare_storage().unwrap_err();
    assert!(matches!(
        error
            .cause()
            .source()
            .unwrap()
            .downcast_ref::<WorkingMemoryError>(),
        Some(WorkingMemoryError::ExecutionFenced)
    ));
    let (pool, held) = retire_request(&state);
    drop(runtime);
    assert!(matches!(
        diagnostic_pin(&state, &pool),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let mut sequence = error.into_sequence();
    for _ in 0..128 {
        let error = sequence.prepare_storage().unwrap_err();
        assert!(matches!(
            error
                .cause()
                .source()
                .unwrap()
                .downcast_ref::<eredu_core::GenerationError>(),
            Some(eredu_core::GenerationError::StorageNotReady)
        ));
        assert_eq!(pool.payload_used_bytes().unwrap(), held);
        // The only way to recover the sole sequence destroys its previous
        // cause. No second live owning error can be accumulated by this loop.
        sequence = error.into_sequence();
    }
    drop(sequence);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn original_bank_busy_preflight_keeps_the_one_bank_without_retained_error_custody() {
    let (mut runtime, state) = runtime(Mode {
        borrow_bank: true,
        ..Mode::default()
    });
    let error = extract(&mut runtime, 3, &[], false, false, None).unwrap_err();
    assert_eq!(error.kind(), BackendFailureKind::Busy);
    assert_eq!(
        error.source().unwrap().downcast_ref::<Rejection>(),
        Some(&Rejection::Busy)
    );
    assert_eq!(state.borrow().order, vec!["admit", "bind", "extract"]);
    let bank = state
        .borrow()
        .active
        .as_ref()
        .unwrap()
        .owner
        .borrow_mut()
        .take_generation_sequence_bank()
        .unwrap();
    let (pool, held) = retire_request(&state);
    drop(runtime);
    assert!(matches!(
        diagnostic_pin(&state, &pool),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), held);
    drop(bank);
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        0,
        "Busy error still lives"
    );
    drop(error);
}

#[test]
fn original_bank_absence_on_a_capture_only_span_creates_no_sequence_owner() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let root = pool.register_host_storage([(1u32, 64)]).unwrap();
    let source = capture_source();
    let q = quote(&pool, &source);
    let (reservation, run, accepted) = accept(&pool, q);
    let (mut owner, witness) = accepted
        .into_funded_text_span_workspace(&run, &reservation)
        .unwrap();
    assert_eq!(
        owner
            .workspace()
            .text_controls()
            .unwrap()
            .sequence_storage_bytes(),
        0
    );
    let references = Arc::strong_count(reservation.0.inner());
    let held = account(&pool, &reservation);
    let errors = std::array::from_fn::<_, 128, _>(|_| {
        assert!(owner.take_generation_sequence_bank().is_none());
        Rejection::Unavailable.into_backend_failure()
    });
    assert_eq!(Arc::strong_count(reservation.0.inner()), references);
    assert_eq!(account(&pool, &reservation), held);
    drop((witness, owner, reservation, run, root, source));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    drop(errors);
}

#[test]
fn original_bank_nested_core_preparation_error_keeps_both_sources_and_actual_custody() {
    let (mut runtime, state) = runtime(Mode {
        explicit_source: true,
        ..Mode::default()
    });
    let sequence = extract(&mut runtime, 5, &[9], false, false, None).unwrap();
    state
        .borrow()
        .active
        .as_ref()
        .unwrap()
        .run
        .borrow_mut()
        .take();
    let failure = sequence.prepare_storage().unwrap_err();
    let nested_bytes = eredu_core::BackendFailure::source_retention_peak_bytes::<
        eredu_core::RetainedSequencePreparationError,
    >()
    .unwrap()
        + eredu_core::BackendFailure::source_retention_peak_bytes::<WorkingMemoryError>().unwrap();
    assert!(state.borrow().r >= u64::try_from(nested_bytes).unwrap() + 24);
    let error = eredu_core::BackendFailure::new(BackendFailureKind::InvalidSession, failure);
    let preparation = error
        .source()
        .unwrap()
        .downcast_ref::<eredu_core::RetainedSequencePreparationError>()
        .unwrap();
    assert!(matches!(
        preparation
            .cause()
            .source()
            .unwrap()
            .downcast_ref::<WorkingMemoryError>(),
        Some(WorkingMemoryError::ExecutionFenced)
    ));
    let (pool, held) = retire_request(&state);
    drop(runtime);
    assert_eq!(pool.payload_used_bytes().unwrap(), held + 64);
    drop(diagnostic_pin(&state, &pool).unwrap());
    drop(error);
    assert!(matches!(
        diagnostic_pin(&state, &pool),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

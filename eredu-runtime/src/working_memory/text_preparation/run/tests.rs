use super::*;
use crate::working_memory::{MemoryLedger, funding::tests::reservation};

fn control_allowance() -> u64 {
    let pool = crate::working_memory::memory_fixture::host_ledger(u64::MAX, 0).unwrap();
    let admission = crate::working_memory::memory_fixture::host_admission(&pool, 0);
    2 * crate::working_memory::memory_fixture::reservation_bytes(&pool, &admission)
}
fn lock_ledger(capacity: u64) -> MemoryLedger {
    crate::working_memory::memory_fixture::host_ledger(capacity + control_allowance(), 0).unwrap()
}
fn lock_reservation(
    pool: &MemoryLedger,
    bytes: u64,
    capacity: u64,
) -> crate::working_memory::WorkingMemoryReservation {
    let admission = crate::working_memory::memory_fixture::host_admission(pool, bytes);
    pool.reserve_with_capacity(
        &InferenceExecutionIdentity::default(),
        &admission,
        crate::working_memory::memory_fixture::resolved_host_limits(
            pool,
            capacity + control_allowance(),
        ),
    )
    .unwrap()
}
fn effective_payload_limit(pool: &MemoryLedger) -> u64 {
    pool.payload_effective_capacity().unwrap() - control_allowance()
}

#[test]
fn reserved_receipt_identity_rejects_same_account_replaced_preparation_without_retaining_control() {
    let pool = lock_ledger(500);
    let reserved = lock_reservation(&pool, 100, 400);
    let execution = reserved.0.execution.clone();
    let geometry = reserved.geometry();
    let request: InferenceRequest = reserved.into();
    let config = TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                max_new_tokens: Some(1),
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let preparation = request
        .prepare_text(&execution, geometry, config.clone())
        .unwrap();
    drop(request);
    // This private unit tests identity transport only. Genuine issuance and
    // completed/fenced/ordinal checks are exercised through the shared core
    // machine by bounded_prefill::text_run, in both receipt modes.
    let before = Arc::strong_count(preparation.request.preparation.as_ref().unwrap());
    let receipt = InferenceTextStepReceipt {
        authority: ReceiptAuthority::new(preparation.request()),
        attempt: 0,
    };
    let aliases = [receipt.clone(), receipt.clone(), receipt.clone()];
    assert_eq!(
        Arc::strong_count(preparation.request.preparation.as_ref().unwrap()),
        before
    );
    assert!(
        receipt
            .authority
            .matches_request(preparation.request())
            .unwrap()
    );
    let mut replaced = preparation.request.clone();
    replaced.preparation = Some(Arc::new(TextPreparationAuthority {
        config: config.clone(),
        state: ControlMutex::new(TextPreparationState::default()),
    }));
    assert!(!receipt.authority.matches_request(&replaced).unwrap());
    preparation.claim_prompt().unwrap().finish().unwrap();
    preparation.bind_prompt().unwrap();
    preparation
        .claim_sampling(config.clone())
        .unwrap()
        .finish()
        .unwrap();
    preparation
        .request
        .begin_prefill(&execution, geometry)
        .unwrap();
    assert!(
        receipt
            .authority
            .matches_request(preparation.request())
            .unwrap()
    );
    assert!(!receipt.authority.matches_request(&replaced).unwrap());
    drop(replaced);
    drop(preparation);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert_eq!(effective_payload_limit(&pool), 500);
    let next = lock_reservation(&pool, 100, 400);
    let next_execution = next.0.execution.clone();
    let next_request: InferenceRequest = next.into();
    let next = next_request
        .prepare_text(&next_execution, geometry, config)
        .unwrap();
    for old in std::iter::once(&receipt).chain(&aliases) {
        assert!(!old.authority.matches_request(next.request()).unwrap());
    }
}

fn lock_config() -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                max_new_tokens: Some(1),
                ..Default::default()
            },
        )
        .unwrap(),
    )
}
fn lock_preparation(
    pool: &MemoryLedger,
    finite_request_limit: bool,
    context: &TextStepContext,
) -> (InferenceTextPreparation, InferenceExecutionIdentity) {
    let reserved = lock_reservation(pool, 100, 400);
    let execution = reserved.0.execution.clone();
    let geometry = reserved.geometry();
    let request = if finite_request_limit {
        InferenceRequest::from(reserved)
    } else {
        let admission = reserved.admission().clone();
        drop(reserved);
        pool.reserve(&execution, &admission).unwrap().into()
    };
    let preparation = request
        .prepare_text(&execution, geometry, lock_config())
        .unwrap();
    preparation.bind_run(context).unwrap();
    lock_ready(&preparation);
    (preparation, execution)
}
fn lock_ready(preparation: &InferenceTextPreparation) {
    preparation.claim_prompt().unwrap().finish().unwrap();
    preparation.bind_prompt().unwrap();
    preparation
        .claim_sampling(lock_config())
        .unwrap()
        .finish()
        .unwrap();
}
fn lock_context() -> TextStepContext {
    crate::working_memory::original_request::tests::issued_lock_context()
}

#[test]
fn request_control_mutex_abandoned_step_waits_then_fences_before_charge_retirement() {
    use crate::working_memory::control_mutex::tests::wait_for_contention;
    for reserved in [false, true] {
        let context = lock_context();
        let pool = lock_ledger(500);
        let (preparation, _) = lock_preparation(&pool, reserved, &context);
        let step = preparation
            .claim_step(&context, PendingTextInput::Prefill(()))
            .unwrap();
        let authority = preparation.authority();
        let guard = authority.state.lock().unwrap();
        std::thread::scope(|scope| {
            let dropped = scope.spawn(move || drop(step));
            wait_for_contention(&authority.state);
            assert_eq!(guard.run.as_ref().unwrap().active, Some(0));
            assert!(!guard.run.as_ref().unwrap().fenced);
            assert_eq!(pool.payload_used_bytes().unwrap(), 100);
            drop(guard);
            dropped.join().unwrap();
        });
        let guard = authority.state.lock().unwrap();
        assert!(guard.run.as_ref().unwrap().fenced);
        assert_eq!(guard.run.as_ref().unwrap().active, None);
        drop(guard);
        assert!(matches!(
            preparation.claim_step(&context, PendingTextInput::Prefill(())),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        drop(preparation);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        assert_eq!(effective_payload_limit(&pool), 500);
    }
}

#[test]
fn request_control_mutex_start_contention_preserves_one_canonical_preparation() {
    use crate::working_memory::control_mutex::tests::wait_for_contention;
    for finite_request_limit in [false, true] {
        let pool = lock_ledger(500);
        let reserved = lock_reservation(&pool, 100, 400);
        let execution = reserved.0.execution.clone();
        let geometry = reserved.geometry();
        let request = if finite_request_limit {
            InferenceRequest::from(reserved)
        } else {
            let admission = reserved.admission().clone();
            drop(reserved);
            pool.reserve(&execution, &admission).unwrap().into()
        };
        let guard = request.start_state().lock().unwrap();
        let preparation = std::thread::scope(|scope| {
            let worker = scope.spawn(|| request.prepare_text(&execution, geometry, lock_config()));
            wait_for_contention(request.start_state());
            assert!(matches!(*guard, RequestStart::Fresh));
            assert_eq!(pool.payload_used_bytes().unwrap(), 100);
            drop(guard);
            worker.join().unwrap().unwrap()
        });
        assert!(matches!(
            request.prepare_text(&execution, geometry, lock_config()),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        ));
        let canonical = request.start_state().lock().unwrap();
        match &*canonical {
            RequestStart::Preparing(value) => {
                assert!(std::ptr::eq(value.as_ref(), preparation.authority()))
            }
            _ => panic!("canonical preparation was replaced"),
        }
        drop(canonical);
        drop(request);
        assert_eq!(pool.payload_used_bytes().unwrap(), 100);
        drop(preparation);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn request_control_mutex_poison_rejects_new_steps_and_retires_exact_request() {
    let context = lock_context();
    for reserved in [false, true] {
        let pool = lock_ledger(500);
        let (preparation, _) = lock_preparation(&pool, reserved, &context);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = preparation.authority().state.lock().unwrap();
            panic!("fixture poisons the actual request preparation state");
        }));
        assert!(panic.is_err());
        assert!(matches!(
            preparation.claim_step(&context, PendingTextInput::Prefill(())),
            Err(WorkingMemoryError::Poisoned)
        ));
        assert_eq!(pool.payload_used_bytes().unwrap(), 100);
        drop(preparation);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        assert_eq!(effective_payload_limit(&pool), 500);
    }
}

#[test]
fn request_control_mutex_opposing_handoffs_keep_order_and_leave_active_runs_unchanged() {
    use crate::working_memory::control_mutex::tests::wait_for_contention;
    let pool = lock_ledger(1000);
    let first = lock_reservation(&pool, 100, 900);
    let execution = first.0.execution.clone();
    let geometry = first.geometry();
    let second = pool.reserve(&execution, &first.0.admission).unwrap();
    let left = InferenceRequest::from(first)
        .prepare_text(&execution, geometry, lock_config())
        .unwrap();
    let right = InferenceRequest::from(second)
        .prepare_text(&execution, geometry, lock_config())
        .unwrap();
    let left_context = lock_context();
    let right_context = lock_context();
    left.bind_run(&left_context).unwrap();
    right.bind_run(&right_context).unwrap();
    lock_ready(&left);
    lock_ready(&right);
    let left_step = left
        .claim_step(&left_context, PendingTextInput::Prefill(()))
        .unwrap();
    let right_step = right
        .claim_step(&right_context, PendingTextInput::Prefill(()))
        .unwrap();
    let left_owner = left.request.preparation.as_ref().unwrap();
    let right_owner = right.request.preparation.as_ref().unwrap();
    let lower = if Arc::as_ptr(left_owner) < Arc::as_ptr(right_owner) {
        left_owner
    } else {
        right_owner
    };
    let hold = lower.state.lock().unwrap();
    std::thread::scope(|scope| {
        let a = scope.spawn(|| left_step.supersede_predecessor(right.request(), &execution));
        let b = scope.spawn(|| right_step.supersede_predecessor(left.request(), &execution));
        wait_for_contention(&lower.state);
        drop(hold);
        assert!(matches!(
            a.join().unwrap(),
            Err(WorkingMemoryError::TextStepActive)
        ));
        assert!(matches!(
            b.join().unwrap(),
            Err(WorkingMemoryError::TextStepActive)
        ));
    });
    for prep in [&left, &right] {
        let state = prep.authority().state.lock().unwrap();
        let run = state.run.as_ref().unwrap();
        assert_eq!(run.active, Some(0));
        assert!(!run.fenced);
        assert!(!run.supersession_claimed);
    }
    assert_eq!(pool.payload_used_bytes().unwrap(), 200);
    drop((left_step, right_step, left, right));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    assert_eq!(effective_payload_limit(&pool), 1000);
}

mod sampling_extension;

mod branch;

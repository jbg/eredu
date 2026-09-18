use super::*;
const CAPACITY: u64 = 1 << 22;
fn original(context: &TextStepContext) -> (WorkingMemoryPool, InferenceTextPreparation) {
    let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
    let reserved = reservation(&pool, 100, CAPACITY);
    let execution = reserved.0.execution.clone();
    let geometry = reserved.geometry();
    let request: InferenceRequest = reserved.into();
    let preparation = request.prepare_text(&execution, geometry, lock_config()).unwrap();
    preparation.bind_run(context).unwrap();
    lock_ready(&preparation);
    (pool, preparation)
}
#[test]
fn branch_gate_rejects_foreign_active_and_same_run_before_planning() {
    let a = lock_context(); let b = lock_context(); let foreign = lock_context();
    let (pool, first) = original(&a); let (_, second) = original(&b);
    let funding = pool.prepare_workspace_metadata(&first.request().memory_reservation().unwrap().0.execution, CAPACITY).unwrap();
    let before = pool.used_bytes().unwrap();
    assert!(matches!(first.request().begin_branch_exchange(&a, first.request(), &a, &funding), Err(WorkingMemoryError::IdentityMismatch)));
    assert!(matches!(first.request().begin_branch_exchange(&foreign, second.request(), &b, &funding), Err(WorkingMemoryError::IdentityMismatch)));
    let step = second.claim_step(&b, PendingTextInput::Prefill(())).unwrap();
    assert!(matches!(first.request().begin_branch_exchange(&a, second.request(), &b, &funding), Err(WorkingMemoryError::TextStepActive)));
    assert_eq!(pool.used_bytes().unwrap(), before);
    step.finish().unwrap();
    assert!(matches!(first.request().begin_branch_exchange(&a, second.request(), &b, &funding), Err(WorkingMemoryError::TextStepOrdinalMismatch { .. })));
}
#[test]
fn branch_gate_preserves_attempts_and_sampler_epoch_and_releases_both_runs() {
    let a = lock_context(); let b = lock_context();
    let (pool, first) = original(&a); let (other_pool, second) = original(&b);
    let funding = pool.prepare_workspace_metadata(&first.request().memory_reservation().unwrap().0.execution, CAPACITY).unwrap();
    let sampling = first.request().begin_sampling_extension(&a, &funding).unwrap().commit().unwrap();
    let gate = first.request().begin_branch_exchange(&a, second.request(), &b, &funding).unwrap();
    gate.validate().unwrap();
    for (preparation, context) in [(&first, &a), (&second, &b)] {
        assert!(matches!(preparation.claim_step(context, PendingTextInput::Prefill(())), Err(WorkingMemoryError::PreparationNotReady)));
        assert!(matches!(preparation.request().begin_sampling_extension(context, &funding), Err(WorkingMemoryError::PreparationNotReady)));
    }
    drop(gate);
    let repeated = first.request().begin_branch_exchange(&a, second.request(), &b, &funding).unwrap();
    repeated.validate().unwrap(); drop(repeated);
    let first_step = first.claim_step(&a, PendingTextInput::Prefill(())).unwrap();
    assert_eq!(sampling.validate_step(&first_step).unwrap(), 0);
    first_step.finish().unwrap();
    let second_step = second.claim_step(&b, PendingTextInput::Prefill(())).unwrap();
    assert!(second_step.original_sampling_is_current().unwrap()); second_step.finish().unwrap();
    drop((sampling, first, second, funding));
    assert_eq!(pool.used_bytes().unwrap(), 0); assert_eq!(other_pool.used_bytes().unwrap(), 0);
}
#[test]
fn branch_metadata_refusal_does_not_claim_either_run() {
    let a = lock_context(); let b = lock_context();
    let (pool, first) = original(&a); let (_, second) = original(&b);
    let funding = pool.prepare_workspace_metadata(&first.request().memory_reservation().unwrap().0.execution, CAPACITY).unwrap();
    funding.reserve_metadata((CAPACITY - pool.used_bytes().unwrap()) as usize).unwrap();
    assert!(matches!(first.request().begin_branch_exchange(&a, second.request(), &b, &funding), Err(WorkingMemoryError::BudgetExceeded { .. })));
    for (preparation, context) in [(&first, &a), (&second, &b)] {
        let state = preparation.authority().state.lock().unwrap();
        let run = state.run.as_ref().unwrap(); assert_eq!(run.control_issue, 0); assert!(run.control_pending.is_none());
        drop(state);
        preparation.claim_step(context, PendingTextInput::Prefill(())).unwrap().finish().unwrap();
    }
}

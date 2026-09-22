use super::*;
const CAPACITY: u64 = 1 << 22;
fn original(context: &TextStepContext) -> (MemoryLedger, InferenceTextPreparation) {
    let pool = crate::working_memory::memory_fixture::host_ledger(CAPACITY, 0).unwrap();
    let reserved = reservation(&pool, 100, CAPACITY);
    let execution = reserved.0.execution.clone();
    let geometry = reserved.geometry();
    let request: InferenceRequest = reserved.into();
    let preparation = request
        .prepare_text(&execution, geometry, lock_config())
        .unwrap();
    preparation.bind_run(context).unwrap();
    lock_ready(&preparation);
    (pool, preparation)
}
fn planning(
    pool: &MemoryLedger,
    preparation: &InferenceTextPreparation,
) -> eredu_core::HostMetadataFunding {
    pool.prepare_workspace_metadata(
        &preparation.request().memory_reservation().0.execution,
        crate::working_memory::memory_fixture::resolved_host_limits(&pool, CAPACITY),
    )
    .unwrap()
}

#[test]
fn sampling_extension_rejects_foreign_active_and_stale_boundaries_before_planning() {
    let context = lock_context();
    let foreign = lock_context();
    let (pool, preparation) = original(&context);
    let funding = planning(&pool, &preparation);
    let baseline = pool.payload_used_bytes().unwrap();
    assert!(matches!(
        preparation
            .request()
            .begin_sampling_extension(&foreign, &funding),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), baseline);
    let step = preparation
        .claim_step(&context, PendingTextInput::Prefill(()))
        .unwrap();
    assert!(matches!(
        preparation
            .request()
            .begin_sampling_extension(&context, &funding),
        Err(WorkingMemoryError::TextStepActive)
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), baseline);
    step.finish().unwrap();
    assert!(matches!(
        preparation
            .request()
            .begin_sampling_extension(&context, &funding),
        Err(WorkingMemoryError::TextStepOrdinalMismatch {
            expected: 1,
            actual: 0
        })
    ));
    assert_eq!(pool.payload_used_bytes().unwrap(), baseline);
    drop((preparation, funding));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn sampling_extension_attempts_are_not_reissued_and_pending_blocks_prediction() {
    let context = lock_context();
    let (pool, preparation) = original(&context);
    let funding = planning(&pool, &preparation);
    let first = preparation
        .request()
        .begin_sampling_extension(&context, &funding)
        .unwrap();
    let abandoned = first.binding.clone();
    assert_eq!(first.remaining_steps(), 1);
    assert!(matches!(
        preparation.claim_step(&context, PendingTextInput::Prefill(())),
        Err(WorkingMemoryError::PreparationNotReady)
    ));
    assert!(matches!(
        preparation.validate_initial_ready_context(&context),
        Err(WorkingMemoryError::PreparationNotReady)
    ));
    drop(first);
    assert!(abandoned.validate_pending().is_err());
    let second = preparation
        .request()
        .begin_sampling_extension(&context, &funding)
        .unwrap();
    assert!(!second.binding.same(&abandoned));
    let active = second.commit().unwrap();
    let third = preparation
        .request()
        .begin_sampling_extension(&context, &funding)
        .unwrap();
    let newest = third.commit().unwrap();
    let step = preparation
        .claim_step(&context, PendingTextInput::Prefill(()))
        .unwrap();
    assert!(!step.original_sampling_is_current().unwrap());
    assert!(active.validate_step(&step).is_err());
    assert_eq!(newest.validate_step(&step).unwrap(), 0);
    assert!(abandoned.validate_step(&step).is_err());
    step.finish().unwrap();
    assert_eq!(
        preparation
            .authority()
            .state
            .lock()
            .unwrap()
            .run
            .as_ref()
            .unwrap()
            .next_attempt,
        1
    );
    drop((active, newest, abandoned, preparation, funding));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn sampling_extension_metadata_refusal_does_not_claim_the_attempt_or_replace_sampling() {
    let context = lock_context();
    let (pool, preparation) = original(&context);
    let funding = planning(&pool, &preparation);
    let remaining = match pool
        .snapshot()
        .unwrap()
        .domains
        .iter()
        .find(|d| d.domain == pool.topology().host_domain())
        .unwrap()
    {
        d => match d.effective_limit {
            eredu_core::MemoryLimit::Finite(limit) => limit - d.current_charge_bytes,
            eredu_core::MemoryLimit::Unlimited => panic!("finite fixture limit"),
        },
    };
    funding.reserve_metadata(remaining as usize).unwrap();
    assert!(matches!(
        preparation
            .request()
            .begin_sampling_extension(&context, &funding),
        Err(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(
        preparation
            .request()
            .sampling_extension_remaining(&context)
            .unwrap(),
        1
    );
    let step = preparation
        .claim_step(&context, PendingTextInput::Prefill(()))
        .unwrap();
    assert!(step.original_sampling_is_current().unwrap());
    step.finish().unwrap();
    drop((preparation, funding));
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

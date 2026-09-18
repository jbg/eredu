use super::*;
use crate::working_memory::{RegisteredSavedSamplingSource, WorkspaceCopyCustody};

// A real aggregate account: its sampler has populated history and its native
// component publishes one registered destination. No native model is asserted
// by this portable scalar/closed-workspace fixture.
fn saved(
    pool: &WorkingMemoryPool,
    adaptive: bool,
) -> (
    FundedSamplerCopy,
    WorkspaceCopyCustody,
    crate::working_memory::WorkingMemoryFundingScope,
    WorkingMemoryStorage<u32>,
) {
    saved_with_capacity(pool, adaptive, 8192)
}

fn saved_with_capacity(
    pool: &WorkingMemoryPool,
    adaptive: bool,
    capacity: u64,
) -> (
    FundedSamplerCopy,
    WorkspaceCopyCustody,
    crate::working_memory::WorkingMemoryFundingScope,
    WorkingMemoryStorage<u32>,
) {
    let physical = pool.register_storage([(1u32, ARRAY_SOURCE_BYTES)]).unwrap();
    let (mut original, preparation, run) = source(pool, capacity, 8, adaptive);
    grow(&mut original, &[3, 11, 7, 19, 5]);
    let (sampler, arrays) = pool
        .copy_sampling_components(
            joint(pool, original.borrow_funded()),
            WorkspaceCopyLimits::new(capacity),
        )
        .unwrap();
    let (custody, scope) = arrays.into_parts();
    let registered = scope
        .adopt_storage_individually([(2u32, 16)])
        .unwrap()
        .remove(&2)
        .unwrap();
    drop((original, preparation, run, physical));
    (sampler, custody, scope, registered)
}

fn full(bytes: u64) -> Admission {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (preparation, run, _) = prepared_request_with_bytes(&pool, u64::MAX, 3, false, bytes);
    let admission = preparation
        .request()
        .memory_reservation()
        .unwrap()
        .admission()
        .clone();
    drop((preparation, run));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    admission
}

fn reserve(
    pool: &WorkingMemoryPool,
    source: &RegisteredSavedSamplingSource<'_, u32>,
    bytes: u64,
    capacity: u64,
) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
    pool.reserve_saved_source_with_capacity_handoff(
        &InferenceExecutionIdentity::default(),
        &full(bytes),
        capacity,
        &[],
        source,
    )
}

#[test]
fn full_exact_and_one_short_preserve_populated_standard_and_adaptive_source() {
    for adaptive in [false, true] {
        let pool = WorkingMemoryPool::new(8192, 0).unwrap();
        let (sampler, custody, scope, registered) = saved(&pool, adaptive);
        scope.certify().unwrap();
        let witness = custody
            .bind_saved_sampling_source(&sampler, registered.clone())
            .unwrap();
        let before = usage(&pool);
        let copies = attempts();
        let source_pointer = history(sampler.as_sampler()).as_ptr();
        let bytes = 137;
        assert!(matches!(
            reserve(&pool, &witness, bytes, before.0 + bytes - 1),
            Err(WorkingMemoryError::BudgetExceeded {
                required_bytes: 137,
                available_bytes: 136
            })
        ));
        assert_eq!(usage(&pool), before);
        let reservation = reserve(&pool, &witness, bytes, before.0 + bytes).unwrap();
        assert_eq!(pool.used_bytes().unwrap(), before.0 + bytes);
        assert_eq!(reservation.admission().incremental_required_bytes, bytes);
        assert!(reservation.requires_funding_scope());
        assert_eq!(attempts(), copies);
        assert_eq!(history(sampler.as_sampler()), &[3, 11, 7, 19, 5]);
        assert_eq!(history(sampler.as_sampler()).as_ptr(), source_pointer);
        drop(reservation);
        assert_eq!(pool.used_bytes().unwrap(), before.0);
        drop(witness);
        drop((registered, sampler, custody));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn same_pool_different_account_and_foreign_registration_cannot_bind() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (sampler, custody, scope, registered) = saved(&pool, false);
    scope.certify().unwrap();
    let independent = pool
        .copy_sampler(sampler.borrow_funded(), SamplerCopyLimits::new(8192))
        .unwrap();
    let before = usage(&pool);
    assert!(matches!(
        custody.bind_saved_sampling_source(&independent, registered.clone()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let foreign = WorkingMemoryPool::new(8192, 0).unwrap();
    let foreign_registered = foreign.register_storage([(2u32, 16)]).unwrap();
    assert!(matches!(
        custody.bind_saved_sampling_source(&sampler, foreign_registered),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        pool.pin_registered_storage([(99u32, 16)]),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let witness = custody
        .bind_saved_sampling_source(&sampler, registered.clone())
        .unwrap();
    assert!(matches!(
        reserve(&foreign, &witness, 64, 8192),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(usage(&pool), before);
    assert_eq!(foreign.used_bytes().unwrap(), 0);
}

#[test]
fn quarantined_pair_after_binding_rejects_before_commit_or_copy() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (sampler, custody, native, registered) = saved(&pool, false);
    let witness = custody
        .bind_saved_sampling_source(&sampler, registered)
        .unwrap();
    // The provider has not certified this scope. A stored witness is not a
    // completion certificate and cannot hide a later failure of either owner.
    drop(native);
    let before = usage(&pool);
    let copies = attempts();
    assert!(matches!(
        reserve(&pool, &witness, 64, 8192),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    assert_eq!(usage(&pool), before);
    assert_eq!(attempts(), copies);
}

#[test]
fn uncopied_complete_source_origin_is_revalidated_at_commit() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (sampler, custody, native, registered) = saved(&pool, false);
    native.certify().unwrap();
    let (other_preparation, other_run, _) = prepared_request_with_bytes(&pool, 8192, 3, false, 80);
    let other_work = other_run.scope().unwrap();
    let other_root = other_work
        .adopt_storage_individually([(77u32, 32)])
        .unwrap();
    let complete = pool
        .pin_registered_storage([(2u32, 16), (77u32, 32)])
        .unwrap();
    let witness = custody
        .bind_saved_sampling_source(&sampler, complete)
        .unwrap();
    drop(other_work);
    let before = usage(&pool);
    assert!(matches!(
        reserve(&pool, &witness, 64, 8192),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    assert_eq!(usage(&pool), before);
    drop((other_root, other_preparation, other_run, registered));
}

#[test]
fn source_pins_follow_funding_scopes_through_success_and_quarantine() {
    for certify in [true, false] {
        let pool = WorkingMemoryPool::new(8192, 0).unwrap();
        let (sampler, custody, native, registered) = saved(&pool, false);
        native.certify().unwrap();
        let witness = custody
            .bind_saved_sampling_source(&sampler, registered.clone())
            .unwrap();
        let reservation = reserve(&pool, &witness, 64, 8192).unwrap();
        let (reservation, run) = reservation.into_funding().unwrap();
        let scope = run.scope().unwrap();
        drop(witness);
        drop((registered, sampler, custody, reservation, run));
        assert_eq!(pool.used_bytes().unwrap(), 64 + 16);
        if certify {
            scope.certify().unwrap();
            assert_eq!(pool.used_bytes().unwrap(), 0);
        } else {
            drop(scope);
            assert_eq!(pool.used_bytes().unwrap(), 64 + 16);
            assert!(pool.pin_registered_storage([(2u32, 16)]).is_ok());
        }
    }
}

#[test]
fn incomplete_or_understated_full_quote_never_uses_source_as_credit() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (sampler, custody, native, registered) = saved(&pool, false);
    native.certify().unwrap();
    let witness = custody
        .bind_saved_sampling_source(&sampler, registered)
        .unwrap();
    let before = usage(&pool);
    let mut admission = full(137);
    admission.incremental_required_bytes = 136;
    assert!(matches!(
        pool.reserve_saved_source_with_capacity_handoff(
            &InferenceExecutionIdentity::default(),
            &admission,
            8192,
            &[],
            &witness
        ),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    admission.incremental_required_bytes = 137;
    admission
        .state
        .execution_workspace
        .as_mut()
        .unwrap()
        .activations = WorkspaceBound::Unknown {
        reason: "missing actual mechanism".into(),
    };
    assert!(matches!(
        pool.reserve_saved_source_with_capacity_handoff(
            &InferenceExecutionIdentity::default(),
            &admission,
            8192,
            &[],
            &witness
        ),
        Err(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(usage(&pool), before);
}

#[test]
fn budget_failure_cannot_commit_capacity_handoff_and_success_keeps_it() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (sampler, custody, native, registered) = saved(&pool, false);
    native.certify().unwrap();
    let source_bytes = pool.used_bytes().unwrap();
    let old_capacity = source_bytes + 64;
    let (old_preparation, mut old_run, _) =
        prepared_request_with_bytes(&pool, old_capacity, 3, false, 64);
    let handoff = old_run.take_capacity_handoff().unwrap();
    let execution = old_preparation
        .request()
        .memory_reservation()
        .unwrap()
        .0
        .execution
        .clone();
    let old_work = old_run.scope().unwrap();
    let old_root = old_work.adopt_storage_individually([(77u32, 16)]).unwrap();
    old_work.certify().unwrap();
    drop(old_run);
    // The old physical output remains; its closed account may delegate a larger
    // ceiling, but no failed source or capacity check may consume that change.
    let witness = custody
        .bind_saved_sampling_source(&sampler, registered)
        .unwrap();
    let before = usage(&pool);
    let larger = old_capacity + 500;
    assert!(matches!(
        pool.reserve_saved_source_with_capacity_handoff(
            &execution,
            &full(larger),
            larger,
            std::slice::from_ref(&handoff),
            &witness
        ),
        Err(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert_eq!(usage(&pool), before);
    let accepted = pool
        .reserve_saved_source_with_capacity_handoff(
            &execution,
            &full(80),
            larger,
            std::slice::from_ref(&handoff),
            &witness,
        )
        .unwrap();
    assert_eq!(pool.effective_capacity().unwrap(), larger);
    drop(accepted);
    assert_eq!(pool.effective_capacity().unwrap(), larger);
    assert_eq!(
        old_preparation
            .request()
            .memory_reservation()
            .unwrap()
            .admission()
            .incremental_required_bytes,
        64
    );
    drop((old_root, old_preparation));
    assert_eq!(pool.effective_capacity().unwrap(), 8192);
}

#[test]
fn stale_origin_failure_precedes_otherwise_valid_capacity_handoff() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (sampler, custody, native, registered) = saved(&pool, false);
    native.certify().unwrap();
    let (origin_preparation, origin_run, _) =
        prepared_request_with_bytes(&pool, 8192, 3, false, 32);
    let origin_work = origin_run.scope().unwrap();
    let origin_root = origin_work
        .adopt_storage_individually([(78u32, 16)])
        .unwrap();
    let complete = pool
        .pin_registered_storage([(2u32, 16), (78u32, 16)])
        .unwrap();
    let witness = custody
        .bind_saved_sampling_source(&sampler, complete)
        .unwrap();
    let old_capacity = pool.used_bytes().unwrap() + 64;
    let (old_preparation, mut old_run, _) =
        prepared_request_with_bytes(&pool, old_capacity, 3, false, 64);
    let execution = old_preparation
        .request()
        .memory_reservation()
        .unwrap()
        .0
        .execution
        .clone();
    let handoff = old_run.take_capacity_handoff().unwrap();
    let work = old_run.scope().unwrap();
    let old_root = work.adopt_storage_individually([(77u32, 16)]).unwrap();
    work.certify().unwrap();
    drop(old_run);
    drop(origin_work);
    let before = usage(&pool);
    assert!(matches!(
        pool.reserve_saved_source_with_capacity_handoff(
            &execution,
            &full(80),
            old_capacity + 500,
            &[handoff],
            &witness
        ),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    assert_eq!(usage(&pool), before);
    drop((
        origin_root,
        origin_preparation,
        origin_run,
        old_root,
        old_preparation,
        registered,
    ));
}

#[test]
fn full_reservation_overflow_leaves_saved_account_and_history_untouched() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (sampler, custody, native, registered) = saved(&pool, false);
    native.certify().unwrap();
    let witness = custody
        .bind_saved_sampling_source(&sampler, registered)
        .unwrap();
    let before = usage(&pool);
    let mut admission = full(1);
    let workspace = admission.state.execution_workspace.as_mut().unwrap();
    workspace.activations = WorkspaceBound::bounded(u64::MAX, "overflowing full component");
    workspace.retained = WorkspaceBound::bounded(1, "second full component");
    assert!(matches!(
        pool.reserve_saved_source_with_capacity_handoff(
            &InferenceExecutionIdentity::default(),
            &admission,
            8192,
            &[],
            &witness
        ),
        Err(WorkingMemoryError::Overflow)
    ));

    assert_eq!(usage(&pool), before);
    assert_eq!(history(sampler.as_sampler()), &[3, 11, 7, 19, 5]);
}

mod incremental;

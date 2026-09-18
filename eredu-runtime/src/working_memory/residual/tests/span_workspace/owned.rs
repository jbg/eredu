use super::*;
use crate::working_memory::{
    OwnedInferenceSpanWorkspace, SpanWorkspaceOwnerError, WorkingMemoryFundingRun,
};
mod plan_retirement;
mod retirement;

fn accepted(
    pool: &WorkingMemoryPool,
) -> (
    WorkingMemoryReservation,
    WorkingMemoryFundingRun,
    IncrementalInferenceQuote,
) {
    let q = replacement_quote(pool, geometry(), 0)
        .into_incremental()
        .with_span_workspace()
        .unwrap();
    let (r, q) = sealed_plan(pool, &q, 1_000_000).unwrap();
    let (r, run) = r.into_funding().unwrap();
    (r, run, q)
}
fn account(pool: &WorkingMemoryPool, r: &WorkingMemoryReservation) -> (u64, u64, usize) {
    let usage = pool.0.usage.lock().unwrap();
    let state = &usage.funding[&r.0.funding.unwrap()];
    (state.remaining, state.host_held, state.scopes)
}
#[test]
fn full_original_host_peak_is_held_exactly_once_before_any_native_publication() {
    for short in [false, true] {
        let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
        let source = pool.register_storage([(1u32, 64)]).unwrap();
        let (r, run, q) = accepted(&pool);
        let p = q.span_workspace().retention_peak_bytes().unwrap();
        let plan = q.span_workspace().plan().clone();
        let native = run.scope().unwrap();
        // Actual independent published storage consumes the original account's
        // balance; no scalar credit or second host reservation is introduced.
        let pressure = native
            .adopt_storage_individually([(20u32, r.bytes() - p + u64::from(short))])
            .unwrap();
        let before = account(&pool, &r);
        let result = q.into_funded_span_workspace(&run, &r);
        if short {
            let error = result.unwrap_err();
            assert!(
                matches!(error.cause(), WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes } if *required_bytes==p && *available_bytes+1==p)
            );
            assert_eq!(account(&pool, &r), before);
            let (q, _) = error.into_parts();
            assert!(q.span_workspace().plan().same_plan(&plan));
            drop(pressure);
            let (owner, witness) = q.into_funded_span_workspace(&run, &r).unwrap();
            assert!(witness.is_none());
            assert_eq!(account(&pool, &r).1, p);
            drop(owner);
        } else {
            let (owner, witness) = result.unwrap();
            assert!(witness.is_none());
            assert_eq!(account(&pool, &r), (before.0, p, before.2 + 1));
            assert!(matches!(
                native.adopt_storage_individually([(21u32, 1)]),
                Err(WorkingMemoryError::BudgetExceeded {
                    required_bytes: 1,
                    available_bytes: 0
                })
            ));
            assert!(owner.workspace().plan().same_plan(&plan));
            drop((owner, pressure));
        }
        native.certify().unwrap();
        drop((run, r, plan, source));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn earlier_diagnostic_alias_is_the_last_owner_of_full_p_after_run_and_receipt_retire() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let source = pool.register_storage([(1u32, 64)]).unwrap();
    let (r, run, q) = accepted(&pool);
    let p = q.span_workspace().retention_peak_bytes().unwrap();
    let earlier = q.span_workspace().plan().clone();
    let native = run.scope().unwrap();
    let live = native
        .adopt_storage_individually([(22u32, r.bytes() - p)])
        .unwrap();
    let (owner, witness) = q.into_funded_span_workspace(&run, &r).unwrap();
    assert!(witness.is_none());
    let request = InferenceRequest::from(&r);
    owner
        .as_reserved_span_workspace()
        .validate_request(&request)
        .unwrap();
    assert!(owner.workspace().plan().same_plan(&earlier));
    native.certify().unwrap();
    let all = pool.used_bytes().unwrap();
    let records = earlier.records().len();
    drop((owner, request, r, run, source));
    assert_eq!(earlier.strong_owner_count(), 1);
    assert_eq!(earlier.records().len(), records);
    assert_eq!(pool.used_bytes().unwrap(), all - 64);
    // Records remain readable until the same alias that retains their host P
    // retires. Published storage has independent custody and remains charged.
    drop(earlier);
    assert_eq!(pool.used_bytes().unwrap(), all - 64 - p);
    drop(live);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn unrelated_reservation_account_and_duplicate_promotion_preserve_the_original_quote() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let source = pool.register_storage([(1u32, 64)]).unwrap();
    let (r, run, q) = accepted(&pool);
    let (other, other_run, other_q) = accepted(&pool);
    let original = q.span_workspace().plan().clone();
    let before = account(&pool, &r);
    let error = q.into_funded_span_workspace(&run, &other).unwrap_err();
    assert!(matches!(
        error.cause(),
        WorkingMemoryError::IdentityMismatch
    ));
    let (q, _) = error.into_parts();
    assert!(q.span_workspace().plan().same_plan(&original));
    let error = q.into_funded_span_workspace(&other_run, &r).unwrap_err();
    let (q, cause) = error.into_parts();
    assert!(matches!(cause, WorkingMemoryError::IdentityMismatch));
    assert_eq!(account(&pool, &r), before);
    let foreign_pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let foreign_root = foreign_pool.register_storage([(1u32, 64)]).unwrap();
    let (foreign_r, foreign_run, foreign_q) = accepted(&foreign_pool);
    let error = q.into_funded_span_workspace(&foreign_run, &r).unwrap_err();
    assert!(matches!(
        error.cause(),
        WorkingMemoryError::IdentityMismatch
    ));
    let (q, _) = error.into_parts();
    assert_eq!(account(&pool, &r), before);
    drop((foreign_r, foreign_run, foreign_q, foreign_root));
    assert_eq!(foreign_pool.used_bytes().unwrap(), 0);
    let duplicate = q.clone();
    let (owner, _) = q.into_funded_span_workspace(&run, &r).unwrap();
    let held = account(&pool, &r);
    let error = duplicate.into_funded_span_workspace(&run, &r).unwrap_err();
    assert!(matches!(
        error.cause(),
        WorkingMemoryError::IdentityMismatch
    ));
    assert_eq!(account(&pool, &r), held);
    let (duplicate, _) = error.into_parts();
    drop((owner, original, r, run));
    assert!(duplicate.span_workspace().plan().records().len() > 0);
    drop((duplicate, other, other_run, other_q, source));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn concurrent_alias_promotions_create_one_host_scope_and_preserve_the_losing_owner() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let source = pool.register_storage([(1u32, 64)]).unwrap();
    let (r, run, q) = accepted(&pool);
    let p = q.span_workspace().retention_peak_bytes().unwrap();
    let other = q.clone();
    let (a, b) = std::thread::scope(|scope| {
        let a = scope.spawn(|| q.into_funded_span_workspace(&run, &r));
        let b = scope.spawn(|| other.into_funded_span_workspace(&run, &r));
        (a.join().unwrap(), b.join().unwrap())
    });
    assert_eq!(usize::from(a.is_ok()) + usize::from(b.is_ok()), 1);
    assert_eq!(account(&pool, &r).1, p);
    assert_eq!(account(&pool, &r).2, 1);
    let (owner, error) = match (a, b) {
        (Ok((owner, None)), Err(error)) | (Err(error), Ok((owner, None))) => (owner, error),
        _ => panic!("exactly one promotion with no explicit joined sources"),
    };
    assert!(matches!(
        error.cause(),
        WorkingMemoryError::IdentityMismatch
    ));
    drop((owner, r, run, source));
    assert!(
        pool.used_bytes().unwrap() >= p,
        "losing error still owns its attached plan"
    );
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn source_witness_and_compact_receipt_recheck_the_same_explicit_zero_and_nonzero_origins() {
    for source_bytes in [0, 24] {
        for quarantine_before in [false, true] {
            let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
            let root = pool.register_storage([(1u32, 64)]).unwrap();
            let source_q = replacement_quote(&pool, geometry(), 0).into_incremental();
            let (source_r, _) = sealed_plan(&pool, &source_q, 1_000_000).unwrap();
            let (source_r, source_run) = source_r.into_funding().unwrap();
            let source_scope = source_run.scope().unwrap();
            let registered = source_scope
                .adopt_storage_individually([(30u32, source_bytes)])
                .unwrap()
                .into_values()
                .next()
                .unwrap();
            let initial = replacement_quote(&pool, geometry(), 0)
                .into_incremental()
                .with_registered_sources(registered.clone())
                .unwrap()
                .with_span_workspace()
                .unwrap();
            let (r, q) = sealed_plan(&pool, &initial, 1_000_000).unwrap();
            drop((initial, registered));
            let (r, run) = r.into_funding().unwrap();
            let plan = q.span_workspace().plan().clone();
            if quarantine_before {
                drop(source_scope);
                let before = account(&pool, &r);
                let error = q.into_funded_span_workspace(&run, &r).unwrap_err();
                assert!(matches!(error.cause(), WorkingMemoryError::ExecutionFenced));
                assert_eq!(account(&pool, &r), before);
                let (q, _) = error.into_parts();
                assert!(q.span_workspace().plan().same_plan(&plan));
                drop(q);
            } else {
                let (owner, witness) = q.into_funded_span_workspace(&run, &r).unwrap();
                let witness = witness.unwrap();
                witness.validate(&pool).unwrap();
                assert_eq!(
                    pool.pin_registered_storage([(30u32, source_bytes)])
                        .unwrap()
                        .bytes(),
                    source_bytes
                );
                let foreign = WorkingMemoryPool::new(1_000_000, 0).unwrap();
                assert!(matches!(
                    witness.validate(&foreign),
                    Err(WorkingMemoryError::IdentityMismatch)
                ));
                drop(source_scope);
                assert!(matches!(
                    witness.validate(&pool),
                    Err(WorkingMemoryError::ExecutionFenced)
                ));
                let receipt = owner.as_reserved_span_workspace();
                let usage = pool.0.usage.lock().unwrap();
                assert!(matches!(
                    receipt.validate_sources(&pool, &usage),
                    Err(WorkingMemoryError::ExecutionFenced)
                ));
                drop(usage);
                drop(receipt);
                drop((owner, witness));
            }
            drop((plan, r, run, source_run, source_r, source_q, root));
            // An intentionally abandoned source scope stays quarantined. This
            // host conversion never certifies or releases that native origin.
            assert!(pool.used_bytes().unwrap() > 0);
        }
    }
}
#[test]
fn unsealed_candidate_rejects_without_attachment_or_losing_diagnostic_aliases() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let initial = replacement_quote(&pool, geometry(), 0).into_incremental();
    let (r, q) = sealed_plan(&pool, &initial, 1_000_000).unwrap();
    drop(initial);
    let (r, run) = r.into_funding().unwrap();
    let plan = q.span_workspace().plan().clone();
    let before = pool.used_bytes().unwrap();
    let error = q.into_funded_span_workspace(&run, &r).unwrap_err();
    assert!(matches!(
        error.cause(),
        WorkingMemoryError::IdentityMismatch
    ));
    assert_eq!(pool.used_bytes().unwrap(), before);
    let (q, _) = error.into_parts();
    assert!(q.span_workspace().plan().same_plan(&plan));
    drop((q, plan, r, run, root));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn retained_peak_contains_actual_attachment_compact_owner_and_failure_moves_before_admission() {
    use std::mem::size_of;
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let q = replacement_quote(&pool, geometry(), 0).into_incremental();
    let p = q.span_workspace().retention_peak_bytes().unwrap();
    let minimum = q.span_workspace().plan().capacity_bytes().unwrap()
        + 3 * (size_of::<OwnedInferenceSpanWorkspace>()
            + size_of::<SpanWorkspaceOwnerError>()
            + size_of::<crate::working_memory::funding::SpanHostCustody>()) as u64;
    assert!(p > minimum);
    let before = q.incremental_bytes();
    let q = q.with_span_workspace().unwrap();
    assert_eq!(q.incremental_bytes(), before + p);
    let exact = 64 + q.incremental_bytes();
    assert!(matches!(
        sealed_plan(&pool, &q, exact - 1),
        Err(PrefillPlanningError::Reservation(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    ));
    let (r, _) = sealed_plan(&pool, &q, exact).unwrap();
    assert_eq!(r.bytes(), before + p);
    drop((q, r, root));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn unwinding_compact_owner_cannot_retire_p_while_an_earlier_plan_alias_survives() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let root = pool.register_storage([(1u32, 64)]).unwrap();
    let (r, run, q) = accepted(&pool);
    let p = q.span_workspace().retention_peak_bytes().unwrap();
    let alias = q.span_workspace().plan().clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (owner, witness) = q.into_funded_span_workspace(&run, &r).unwrap();
        assert!(witness.is_none());
        assert_eq!(owner.workspace().plan().records().len(), 3);
        panic!("after successful host-only promotion");
    }));
    assert!(result.is_err());
    assert_eq!(account(&pool, &r).1, p);
    assert_eq!(account(&pool, &r).2, 1);
    drop((r, run, root));
    assert!(pool.used_bytes().unwrap() >= p);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

use super::*;
use crate::working_memory::{InferenceSpanWorkspacePlan, InferenceWorkspaceSpan};
use std::sync::{Arc, Barrier};

fn snapshot(plan: &InferenceSpanWorkspacePlan) -> Vec<(InferenceWorkspaceSpan, Option<u64>)> {
    plan.records()
        .iter()
        .map(|record| (record.span().clone(), record.new_allocation_bytes()))
        .collect()
}
fn assert_records(
    plan: &InferenceSpanWorkspacePlan,
    expected: &[(InferenceWorkspaceSpan, Option<u64>)],
) {
    assert_eq!(plan.records().len(), expected.len());
    for (record, (span, bytes)) in plan.records().iter().zip(expected) {
        assert_eq!(record.span(), span);
        assert_eq!(&record.new_allocation_bytes(), bytes);
    }
}

#[test]
fn original_plan_concurrent_final_aliases_keep_records_and_one_original_hold() {
    for _ in 0..4 {
        let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
        let source = pool.register_host_storage([(1u32, 64)]).unwrap();
        let (reservation, run, quote) = accepted(&pool);
        let held = quote.span_workspace().retention_peak_bytes().unwrap();
        let plan = quote.span_workspace().plan().clone();
        let expected = Arc::new(snapshot(&plan));
        assert!(!expected.is_empty());
        let pointer = plan.records().as_ptr() as usize;
        let (owner, witness) = quote
            .into_funded_span_workspace(&run, &reservation)
            .unwrap();
        assert!(witness.is_none());
        assert!(owner.workspace().plan().same_plan(&plan));
        assert_eq!(account(&pool, &reservation).1, held);
        assert_eq!(account(&pool, &reservation).2, 1);
        drop((owner, reservation, run, source));
        assert_eq!(plan.strong_owner_count(), 1);
        assert_eq!(pool.payload_used_bytes().unwrap(), held);
        let ready = Arc::new(Barrier::new(5));
        let release = Arc::new(Barrier::new(5));
        std::thread::scope(|scope| {
            let workers = (0..4)
                .map(|_| {
                    let alias = plan.clone();
                    let same = alias.same_plan(&plan);
                    let expected = Arc::clone(&expected);
                    let ready = Arc::clone(&ready);
                    let release = Arc::clone(&release);
                    scope.spawn(move || {
                        // Observe before rendezvous, but preserve a callback
                        // panic as data so every peer still reaches release.
                        let before = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            (snapshot(&alias), alias.records().as_ptr() as usize)
                        }));
                        ready.wait();
                        release.wait();
                        let after = snapshot(&alias);
                        drop(alias);
                        (same, before, after, expected)
                    })
                })
                .collect::<Vec<_>>();
            drop(plan);
            ready.wait();
            let observed_held = pool.payload_used_bytes();
            release.wait();
            for worker in workers {
                let (same, before, after, expected) = worker.join().unwrap();
                let (before, actual_pointer) = before.unwrap();
                assert!(same);
                assert_eq!(before, *expected);
                assert_eq!(actual_pointer, pointer);
                assert_eq!(after, *expected);
            }
            assert_eq!(observed_held.unwrap(), held);
        });
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

#[test]
fn original_plan_alias_survives_promoted_owner_unwind_with_exact_records_and_charge() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let source = pool.register_host_storage([(1u32, 64)]).unwrap();
    let (reservation, run, quote) = accepted(&pool);
    let held = quote.span_workspace().retention_peak_bytes().unwrap();
    let plan = quote.span_workspace().plan().clone();
    let expected = snapshot(&plan);
    let pointer = plan.records().as_ptr();
    let geometry = plan.geometry();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (owner, witness) = quote
            .into_funded_span_workspace(&run, &reservation)
            .unwrap();
        assert!(witness.is_none());
        assert!(owner.workspace().plan().same_plan(&plan));
        assert_records(owner.workspace().plan(), &expected);
        panic!("after actual original plan promotion");
    }))
    .unwrap_err();
    assert_eq!(
        result.downcast_ref::<&str>(),
        Some(&"after actual original plan promotion")
    );
    assert_eq!(account(&pool, &reservation).1, held);
    assert_eq!(account(&pool, &reservation).2, 1);
    drop((reservation, run, source));
    assert_eq!(plan.strong_owner_count(), 1);
    assert_eq!(pool.payload_used_bytes().unwrap(), held);
    assert_eq!(plan.geometry(), geometry);
    assert_eq!(plan.records().as_ptr(), pointer);
    assert_records(&plan, &expected);
    drop(plan);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn original_plan_late_source_failure_keeps_readable_records_and_original_hold() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1_000_000, 0).unwrap();
    let root = pool.register_host_storage([(1u32, 64)]).unwrap();
    let source_quote =
        replacement_quote(&pool, geometry(), publication_controls()).into_incremental();
    let (source_reservation, _) = sealed_plan(&pool, &source_quote, 1_000_000).unwrap();
    let (source_reservation, source_run) = source_reservation.into_funding().unwrap();
    let source_scope = source_run.scope().unwrap();
    let registered = source_scope
        .adopt_host_storage_individually([(30u32, 24)])
        .unwrap()
        .into_values()
        .next()
        .unwrap();
    // Keep this real source registration outside the failed quote so removing
    // the last plan changes only its original host charge, not source backing.
    let initial = replacement_quote(&pool, geometry(), 0)
        .into_incremental()
        .with_registered_sources(registered.clone())
        .unwrap()
        .with_span_workspace()
        .unwrap();
    let (reservation, quote) = sealed_plan(&pool, &initial, 1_000_000).unwrap();
    drop(initial);
    let (reservation, run) = reservation.into_funding().unwrap();
    let held = quote.span_workspace().retention_peak_bytes().unwrap();
    let plan = quote.span_workspace().plan().clone();
    let expected = snapshot(&plan);
    let pointer = plan.records().as_ptr();
    // Existing failure-only hook drops the actual native source scope after
    // the once-only host attachment and before its final health validation.
    crate::working_memory::funding::quarantine_source_after_next_attachment(source_scope);
    let error = quote
        .into_funded_span_workspace(&run, &reservation)
        .unwrap_err();
    assert!(matches!(error.cause(), WorkingMemoryError::ExecutionFenced));
    assert_eq!(account(&pool, &reservation).1, held);
    assert_eq!(account(&pool, &reservation).2, 1);
    let (quote, cause) = error.into_parts();
    assert!(matches!(cause, WorkingMemoryError::ExecutionFenced));
    assert!(quote.span_workspace().plan().same_plan(&plan));
    assert_records(quote.span_workspace().plan(), &expected);
    drop((quote, reservation, run));
    assert_eq!(plan.strong_owner_count(), 1);
    let before = pool.payload_used_bytes().unwrap();
    assert!(before >= held);
    assert_eq!(plan.records().as_ptr(), pointer);
    assert_records(&plan, &expected);
    drop(plan);
    assert_eq!(pool.payload_used_bytes().unwrap(), before - held);
    drop((
        registered,
        source_reservation,
        source_run,
        source_quote,
        root,
    ));
    assert!(
        pool.payload_used_bytes().unwrap() > 0,
        "original source quarantine survives"
    );
}

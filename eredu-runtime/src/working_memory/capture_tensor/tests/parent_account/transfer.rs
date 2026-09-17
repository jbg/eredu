use super::*;
use std::error::Error as _;

#[test]
fn exact_parent_and_native_share_pressure_with_full_source_pin() {
    let source = ordinary();
    let p = plan(&source).initialization_peak_bytes();
    for available in [p - 1, p + 73] {
        let pool = WorkingMemoryPool::new(90 + p + 73, 0).unwrap();
        let storage = pool.register_storage([(7u32, 90)]).unwrap();
        let (reservation, run) = fresh(&pool, available);
        let mut native = run.scope().unwrap();
        let before = ledger(&pool);
        let allocations = ALLOCATIONS.get();
        let result = run.prepare_capture_tensor_with_source(
            &reservation,
            &mut native,
            plan(&source),
            storage.clone(),
        );
        if available < p {
            assert!(matches!(
                result,
                Err(CaptureTensorConstructionError::Memory(
                    WorkingMemoryError::BudgetExceeded { .. }
                ))
            ));
            assert_eq!(before, ledger(&pool));
            assert_eq!(ALLOCATIONS.get(), allocations);
            drop(result);
        } else {
            let mut builder = result.unwrap();
            assert_eq!(builder.protected_bytes(), p);
            let adoption = run.scope().unwrap();
            assert!(matches!(
                adoption.adopt_storage_individually([(8u32, 74)]),
                Err(WorkingMemoryError::BudgetExceeded { .. })
            ));
            let produced = adoption.adopt_storage_individually([(8u32, 73)]).unwrap();
            for n in 0..12 {
                builder.push_f32(n as f32 + 0.5).unwrap();
            }
            let observation = builder.finish().unwrap();
            adoption.certify().unwrap();
            let alias = observation.clone();
            drop((observation, storage, reservation, run));
            assert_eq!(pool.used_bytes().unwrap(), 90 + p + 73);
            native.certify().unwrap();
            assert_eq!(pool.used_bytes().unwrap(), p + 73);
            assert_eq!(alias.shape(), &[3, 4]);
            drop(alias);
            assert_eq!(pool.used_bytes().unwrap(), 73);
            drop(produced);
            assert_eq!(pool.used_bytes().unwrap(), 0);
            continue;
        }
        native.certify().unwrap();
        drop((storage, reservation, run));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn unrelated_scope_parent_and_source_domain_reject_without_mutation() {
    let source = ordinary();
    let p = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(4 * p + 18, 0).unwrap();
    let foreign = WorkingMemoryPool::new(p + 18, 0).unwrap();
    let local = pool.register_storage([(1u32, 9)]).unwrap();
    let other_storage = foreign.register_storage([(1u32, 9)]).unwrap();
    let (reservation, run) = fresh(&pool, p);
    let (other, other_run) = fresh(&pool, p);
    let mut native = run.scope().unwrap();
    let mut other_native = other_run.scope().unwrap();
    let before = ledger(&pool);
    let allocated = ALLOCATIONS.get();
    assert!(run
        .prepare_capture_tensor_with_source(
            &reservation,
            &mut other_native,
            plan(&source),
            local.clone()
        )
        .is_err());
    assert!(run
        .prepare_capture_tensor_with_source(&other, &mut native, plan(&source), local.clone())
        .is_err());
    assert!(run
        .prepare_capture_tensor_with_source(
            &reservation,
            &mut native,
            plan(&source),
            other_storage.clone()
        )
        .is_err());
    assert_eq!(ledger(&pool), before);
    assert_eq!(ALLOCATIONS.get(), allocated);
    native.certify().unwrap();
    other_native.certify().unwrap();
}

#[test]
fn allocation_unwind_restores_prior_scope_inventory_and_hold() {
    let source = ordinary();
    let p = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(2 * p + 30, 0).unwrap();
    let a = pool.register_storage([(1u32, 11)]).unwrap();
    let b = pool.register_storage([(2u32, 19)]).unwrap();
    let (reservation, run) = fresh(&pool, p);
    let mut native = run.scope().unwrap();
    drop(
        run.prepare_capture_tensor_with_source(&reservation, &mut native, plan(&source), a.clone())
            .unwrap(),
    );
    let before = ledger(&pool);
    PANIC_BEFORE_BUFFERS.set(true);
    assert!(catch_unwind(AssertUnwindSafe(|| {
        drop(run.prepare_capture_tensor_with_source(
            &reservation,
            &mut native,
            plan(&source),
            b.clone(),
        ));
    }))
    .is_err());
    assert_eq!(ledger(&pool), before);
    drop((a, b));
    assert_eq!(pool.used_bytes().unwrap(), p + 11); // B was rolled back, A remains.
    drop((reservation, run));
    native.certify().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn finished_host_and_abandoned_scope_keep_distinct_custody_even_after_prior_quarantine() {
    let source = ordinary();
    let p = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(p + 30, 0).unwrap();
    let a = pool.register_storage([(1u32, 11)]).unwrap();
    let b = pool.register_storage([(2u32, 19)]).unwrap();
    let (reservation, run) = fresh(&pool, p);
    let mut early = run.scope().unwrap();
    let mut later = run.scope().unwrap();
    drop(
        run.prepare_capture_tensor_with_source(&reservation, &mut early, plan(&source), a.clone())
            .unwrap(),
    );
    drop(
        run.prepare_capture_tensor_with_source(&reservation, &mut later, plan(&source), a.clone())
            .unwrap(),
    );
    let mut builder = run
        .prepare_capture_tensor_with_source(&reservation, &mut later, plan(&source), b.clone())
        .unwrap();
    for n in 0..12 {
        builder.push_f32(n as f32).unwrap();
    }
    let host = builder.finish().unwrap();
    drop((a, b, early)); // first quarantine has only A
    drop(host); // host never owns or refunds B's native scope pin
    drop((later, reservation, run));
    assert_eq!(pool.used_bytes().unwrap(), p + 30);
    assert!(pool.acquire_unquoted().is_err());
}

#[test]
fn source_origin_health_and_closed_parent_are_rechecked_during_fill_and_finish() {
    for close_parent in [false, true] {
        let source = ordinary();
        let p = plan(&source).initialization_peak_bytes();
        let pool = WorkingMemoryPool::new(p + 31, 0).unwrap();
        let (source_res, source_run) = fresh(&pool, 31);
        let source_native = source_run.scope().unwrap();
        let registered = source_native
            .adopt_storage_individually([(1u32, 31)])
            .unwrap()
            .into_values()
            .next()
            .unwrap();
        let (reservation, run) = fresh(&pool, p);
        let mut native = run.scope().unwrap();
        let mut builder = run
            .prepare_capture_tensor_with_source(
                &reservation,
                &mut native,
                plan(&source),
                registered,
            )
            .unwrap();
        builder.push_f32(7.).unwrap();
        let mut run = Some(run);
        if close_parent {
            drop(run.take());
            source_native.certify().unwrap();
        } else {
            drop(source_native);
        }
        assert!(matches!(
            builder.push_f32(8.),
            Err(WorkingMemoryError::ExecutionFenced)
        ));
        let error = builder.finish().unwrap_err();
        assert!(error
            .source()
            .unwrap()
            .downcast_ref::<WorkingMemoryError>()
            .is_some());
        assert!(error.into_builder().is_err());
        drop((native, reservation, source_res, source_run, run));
    }
}

#[test]
fn incomplete_finish_keeps_same_buffers_and_does_not_certify_native_scope() {
    let source = ordinary();
    let p = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(p + 9, 0).unwrap();
    let storage = pool.register_storage([(1u32, 9)]).unwrap();
    let (reservation, run) = fresh(&pool, p);
    let mut native = run.scope().unwrap();
    let mut builder = run
        .prepare_capture_tensor_with_source(&reservation, &mut native, plan(&source), storage)
        .unwrap();
    builder.push_f32(1.5).unwrap();
    let mut builder = builder.finish().unwrap_err().into_builder().unwrap();
    assert_eq!(builder.initialized_count(), 1);
    for _ in 1..12 {
        builder.push_f32(2.5).unwrap();
    }
    let host = builder.finish().unwrap();
    drop(host);
    drop((run, reservation, native));
    assert_eq!(pool.used_bytes().unwrap(), p + 9);
}

#[test]
fn exact_borrowed_scope_keeps_source_after_a_sibling_certifies() {
    let source = ordinary();
    let p = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(p + 19, 0).unwrap();
    let storage = pool.register_storage([(1u32, 19)]).unwrap();
    let (reservation, run) = fresh(&pool, p);
    let sibling = run.scope().unwrap();
    let mut bound = run.scope().unwrap();
    let mut builder = run
        .prepare_capture_tensor_with_source(&reservation, &mut bound, plan(&source), storage)
        .unwrap();
    // Only the unrelated sibling remains accessible while `bound` is borrowed;
    // no validate/fill/finish API accepts a substituting scope argument.
    sibling.certify().unwrap();
    for n in 0..12 {
        builder.push_f32(n as f32).unwrap();
    }
    let host = builder.finish().unwrap();
    drop(host);
    drop((reservation, run));
    assert_eq!(pool.used_bytes().unwrap(), p + 19);
    bound.certify().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn quarantined_registered_source_rejects_before_capture_allocation() {
    let source = ordinary();
    let p = plan(&source).initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(p + 31, 0).unwrap();
    let (source_res, source_run) = fresh(&pool, 31);
    let source_native = source_run.scope().unwrap();
    let registered = source_native
        .adopt_storage_individually([(1u32, 31)])
        .unwrap()
        .into_values()
        .next()
        .unwrap();
    let (reservation, run) = fresh(&pool, p);
    let mut native = run.scope().unwrap();
    drop(source_native);
    let before = ledger(&pool);
    let allocations = ALLOCATIONS.get();
    assert!(matches!(
        run.prepare_capture_tensor_with_source(
            &reservation,
            &mut native,
            plan(&source),
            registered.clone()
        ),
        Err(CaptureTensorConstructionError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert_eq!(ledger(&pool), before);
    assert_eq!(ALLOCATIONS.get(), allocations);
    native.certify().unwrap();
    drop((registered, source_res, source_run, reservation, run));
    assert_eq!(pool.used_bytes().unwrap(), 31);
}

use super::*;

fn state(run: &WorkingMemoryFundingRun) -> (u64, u64, usize, bool, bool) {
    let usage = run.pool.0.usage.lock().unwrap();
    let s = usage.funding.get(&run.id).unwrap();
    (
        s.domains[1].remaining,
        s.host_held,
        s.scopes,
        s.run_open,
        s.metadata_live,
    )
}

#[test]
fn original_reservation_clones_validate_without_scopes_or_accounting_changes() {
    for bytes in [0, 128] {
        let pool = device_ledger(1024, 0).unwrap();
        let (metadata, run) = device_reservation(&pool, bytes, 1024)
            .into_funding()
            .unwrap();
        let clone = metadata.clone();
        let before = (balances(&pool), state(&run));
        for actual in [&metadata, &clone, &metadata] {
            run.validate_reservation(actual).unwrap();
        }
        assert_eq!((balances(&pool), state(&run)), before);
        drop(metadata);
        run.validate_reservation(&clone).unwrap();
        run.close().unwrap();
        assert_eq!(pool.device_used_bytes().unwrap(), 0);
        // close consumes the only run; no closed run validation can be invoked
        // through the public API, and surviving metadata cannot recreate one.
        blocked(&pool);
        drop(clone);
        crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    }
}

#[test]
fn equal_pool_account_and_foreign_domain_cannot_replace_original_metadata() {
    let pool = device_ledger(1024, 0).unwrap();
    let other = device_ledger(1024, 0).unwrap();
    let (metadata, run) = device_reservation(&pool, 128, 1024).into_funding().unwrap();
    let make = |target: &MemoryLedger| {
        target
            .reserve_with_capacity(
                &metadata.0.execution,
                metadata.admission(),
                device_limits(target, 1024),
            )
            .unwrap()
    };
    let (different, different_run) = make(&pool).into_funding().unwrap();
    let (foreign, foreign_run) = make(&other).into_funding().unwrap();
    let unconverted = make(&pool);
    let before = (balances(&pool), balances(&other), state(&run));
    for actual in [&different, &foreign, &unconverted] {
        assert_eq!(
            run.validate_reservation(actual),
            Err(WorkingMemoryError::IdentityMismatch)
        );
    }
    assert_eq!(
        different_run.validate_reservation(&metadata),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!(
        foreign_run.validate_reservation(&metadata),
        Err(WorkingMemoryError::IdentityMismatch)
    );
    run.validate_reservation(&metadata).unwrap();
    assert_eq!((balances(&pool), balances(&other), state(&run)), before);
    drop((
        run,
        different_run,
        foreign_run,
        metadata,
        different,
        foreign,
        unconverted,
    ));
    assert_eq!(pool.device_used_bytes().unwrap(), 0);
    assert_eq!(other.device_used_bytes().unwrap(), 0);
}

#[test]
fn quarantined_original_account_rejects_without_retiring_its_charge_or_scope() {
    for bytes in [0, 128] {
        let pool = device_ledger(1024, 0).unwrap();
        let (metadata, run) = device_reservation(&pool, bytes, 1024)
            .into_funding()
            .unwrap();
        let abandoned = run.scope().unwrap();
        run.validate_reservation(&metadata).unwrap();
        drop(abandoned);
        let before = (balances(&pool), state(&run));
        for _ in 0..3 {
            assert_eq!(
                run.validate_reservation(&metadata),
                Err(WorkingMemoryError::ExecutionFenced)
            );
        }
        assert_eq!((balances(&pool), state(&run)), before);
        drop((run, metadata));
        assert_eq!(pool.device_used_bytes().unwrap(), bytes);
        blocked(&pool);
    }
}

#[test]
fn poisoned_usage_returns_typed_error_and_cleanup_uses_existing_retirement() {
    let pool = device_ledger(1024, 0).unwrap();
    let (metadata, run) = device_reservation(&pool, 128, 1024).into_funding().unwrap();
    let before = state(&run);
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = pool.0.usage.lock().unwrap();
        panic!("poison usage before read-only validation");
    }));
    assert!(failure.is_err());
    assert_eq!(
        run.validate_reservation(&metadata),
        Err(WorkingMemoryError::Poisoned)
    );
    {
        let usage = pool
            .0
            .usage
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let s = usage.funding.get(&run.id).unwrap();
        assert_eq!(
            (
                s.domains[1].remaining,
                s.host_held,
                s.scopes,
                s.run_open,
                s.metadata_live
            ),
            before
        );
    }
    let id = run.id;
    drop((run, metadata));
    let usage = pool
        .0
        .usage
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    // Poison is global accounting uncertainty. Cleanup must quarantine the
    // existing account before deriving any refund, even without native work.
    assert_eq!(usage.domains[1].reserved, 128);
    let retained = usage
        .funding
        .get(&id)
        .expect("quarantined original account");
    assert!(retained.quarantined);
    assert!(!retained.run_open && !retained.metadata_live);
    assert_eq!(
        (
            retained.domains[1].remaining,
            retained.host_held - retained.control_floor,
            retained.scopes
        ),
        (128, 0, 0)
    );
}

use super::*;
#[test]
fn borrowed_funding_start_preserves_rejected_reservation_and_starts_only_once() {
    let pool = device_ledger(500, 0).unwrap();
    let mut original = device_reservation(&pool, 100, 200);
    let alias = original.clone();
    assert!(matches!(
        original.start_funding(),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(balances(&pool), (100, 0, 100));
    assert_eq!(pool.device_capacity().unwrap(), 200);
    drop(alias);
    let run = original.start_funding().unwrap();
    assert!(matches!(
        original.start_funding(),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(balances(&pool), (100, 0, 100));
    drop(run);
    assert_eq!(pool.device_used_bytes().unwrap(), 0);
    assert_eq!(pool.device_capacity().unwrap(), 200);
    drop(original);
    assert_eq!(pool.device_capacity().unwrap(), 500);

    let mut original = device_reservation(&pool, 100, 200);
    *original.0.start.lock().unwrap() = RequestStart::Started(None);
    assert!(matches!(
        original.start_funding(),
        Err(WorkingMemoryError::AlreadyStarted)
    ));
    assert_eq!(pool.device_used_bytes().unwrap(), 100);
    drop(original);
    assert_eq!(pool.device_used_bytes().unwrap(), 0);
}
#[test]
fn overflow_and_poison_rejections_retain_the_actual_unconverted_charge() {
    let pool = device_ledger(500, 0).unwrap();
    let mut original = device_reservation(&pool, 100, 200);
    let old_next = pool.0.usage.lock().unwrap().next_funding;
    let original_id = original.0.account_id;
    pool.0.usage.lock().unwrap().next_funding = u64::MAX;
    // This conversion consumes the existing node; it issues no second ID.
    let run = original.start_funding().unwrap();
    assert_eq!(original.0.funding, Some(original_id));
    let prior = (
        pool.device_used_bytes().unwrap(),
        pool.device_peak_bytes().unwrap(),
        pool.device_capacity().unwrap(),
    );
    let refused = pool.reserve_with_capacity(
        &original.0.execution,
        original.admission(),
        device_limits(&pool, 200),
    );
    assert!(matches!(refused, Err(WorkingMemoryError::Overflow)));
    assert_eq!(
        (
            pool.device_used_bytes().unwrap(),
            pool.device_peak_bytes().unwrap(),
            pool.device_capacity().unwrap()
        ),
        prior
    );
    assert_eq!(pool.0.usage.lock().unwrap().next_funding, u64::MAX);
    assert_eq!(original.0.funding, Some(original_id));
    drop(run);
    drop(original);
    assert_eq!(pool.device_used_bytes().unwrap(), 0);
    // Restore the exact injected fixture counter, never an already issued ID.
    pool.0.usage.lock().unwrap().next_funding = old_next;
    let mut original = device_reservation(&pool, 100, 200);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _start = original.0.start.lock().unwrap();
        panic!("poison actual request start");
    }));
    assert!(panic.is_err());
    assert!(matches!(
        original.start_funding(),
        Err(WorkingMemoryError::Poisoned)
    ));
    assert_eq!(pool.device_used_bytes().unwrap(), 100);
    drop(original);
    assert_eq!(pool.device_used_bytes().unwrap(), 0);

    let mut original = device_reservation(&pool, 100, 200);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _usage = pool.0.usage.lock().unwrap();
        panic!("poison actual accounting lock");
    }));
    assert!(panic.is_err());
    assert!(matches!(
        original.start_funding(),
        Err(WorkingMemoryError::Poisoned)
    ));
    assert_eq!(
        pool.0.usage.lock().unwrap_err().into_inner().domains[1].reserved,
        100
    );
    drop(original);
    assert_eq!(
        pool.0.usage.lock().unwrap_err().into_inner().domains[1].reserved,
        100
    );
}

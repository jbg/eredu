use super::*;
use std::sync::{
    Arc, Barrier,
    atomic::{AtomicUsize, Ordering},
};

mod construction;

#[test]
fn competing_planning_producers_reserve_before_allocation_and_keep_last_owner_charged() {
    let capacity = 1 << 20;
    let pool = crate::working_memory::memory_fixture::host_ledger(capacity, 17).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let first = pool
        .prepare_workspace_metadata(
            &execution,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        )
        .unwrap();
    let second = pool
        .prepare_workspace_metadata(
            &execution,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        )
        .unwrap();
    let before = pool.payload_used_bytes().unwrap();
    let request = usize::try_from((capacity - before) / 2 + 1).unwrap();
    let start = Arc::new(Barrier::new(3));
    let produced = Arc::new(AtomicUsize::new(0));
    let spawn = |owner: HostMetadataFunding| {
        let start = start.clone();
        let produced = produced.clone();
        std::thread::spawn(move || {
            start.wait();
            let result = owner.reserve_metadata(request).map(|()| {
                produced.fetch_add(1, Ordering::SeqCst);
                vec![127u8; request]
            });
            // Payload precedes custody so its physical backing retires first.
            (result, owner)
        })
    };
    let a = spawn(first.clone());
    let b = spawn(second.clone());
    start.wait();
    let a = a.join().unwrap();
    let b = b.join().unwrap();
    assert_eq!(produced.load(Ordering::SeqCst), 1);
    assert_eq!(a.0.is_ok() as usize + b.0.is_ok() as usize, 1);
    for (result, _) in [&a, &b] {
        match result {
            Ok(bytes) => assert!(bytes.iter().all(|byte| *byte == 127)),
            Err(error) => assert!(matches!(
                error,
                HostMetadataFundingError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required, limit_bytes, existing_bytes, .. })
                    if *required == request as u64 && limit_bytes-existing_bytes < *required
            )),
        }
    }
    assert_eq!(pool.payload_used_bytes().unwrap(), before + request as u64);
    assert!(pool.acquire_unquoted().is_err());
    drop(first);
    drop(second);
    // Thread-returned owners keep both the failed candidate's fixed controls
    // and the successful candidate's actual destination charged.
    assert_eq!(pool.payload_used_bytes().unwrap(), before + request as u64);
    drop(a);
    drop(b);
    assert_eq!(pool.payload_used_bytes().unwrap(), 17);
}

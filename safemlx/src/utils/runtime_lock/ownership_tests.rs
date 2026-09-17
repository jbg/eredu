use super::*;
use crate::utils::allocation_test::measure;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

fn await_flag(flag: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !flag() {
        assert!(
            Instant::now() < deadline,
            "runtime test handshake timed out"
        );
        std::thread::yield_now();
    }
}

#[test]
fn private_runtime_first_contention_and_final_release_do_not_allocate() {
    // Root also runs this case alone in a fresh test process. No inference,
    // allocator warmup or other runtime-lock acquisition precedes this body.
    // The measured acquire is the actual helper used by ordinary enter; its
    // deliberately ordinary reclaim/hook work remains outside this claim.
    let start = AtomicBool::new(false);
    let acquired = AtomicBool::new(false);
    waiting::OBSERVED.store(0, Ordering::SeqCst);
    std::thread::scope(|threads| {
        let worker = threads.spawn(|| {
            await_flag(|| start.load(Ordering::SeqCst));
            let ((), allocations) = measure(|| {
                let guard = acquire();
                acquired.store(true, Ordering::SeqCst);
                drop(guard);
            });
            allocations
        });
        let ((), allocations) = measure(|| {
            let guard = try_enter_for_recovery().unwrap();
            start.store(true, Ordering::SeqCst);
            // Set only after the real pause worker has performed its first
            // sleep, not merely because time happened to elapse.
            await_flag(|| waiting::OBSERVED.load(Ordering::SeqCst) == 3);
            assert!(!acquired.load(Ordering::SeqCst));
            drop(guard);
        });
        assert_eq!(allocations, 0, "owner acquisition/final release allocated");
        assert_eq!(worker.join().unwrap(), 0, "contended acquisition allocated");
        assert!(acquired.load(Ordering::SeqCst));
    });
}

#[test]
fn private_runtime_recursive_and_recovery_owners_exclude_foreign_entry_until_final_drop() {
    let outer = enter();
    let inner = try_enter().unwrap();
    assert!(!can_reclaim_submission_resources());
    try_retire(|| {
        let nested = enter();
        assert!(!can_reclaim_submission_resources());
        drop(nested);
    })
    .unwrap();
    let foreign_busy = || {
        std::thread::spawn(|| try_enter_for_recovery().is_none())
            .join()
            .unwrap()
    };
    assert!(foreign_busy());
    drop(inner);
    assert!(foreign_busy());
    drop(outer);
    assert!(can_reclaim_submission_resources());
    assert!(!foreign_busy());
}

#[test]
fn private_runtime_unwind_releases_real_owner_without_poison_or_hooks_during_recovery() {
    let error = std::panic::catch_unwind(|| {
        let _guard = enter();
        let _ = try_retire(|| {
            let _nested = enter();
            panic!("runtime owner unwind probe");
        });
    });
    assert!(error.is_err());
    assert!(can_reclaim_submission_resources());
    assert!(std::thread::spawn(|| try_enter_for_recovery().is_some())
        .join()
        .unwrap());
    // The retiring flag must have been restored by its existing Drop reset.
    let _guard = enter();
    assert!(!RETIRING_SUBMISSIONS.with(Cell::get));
}

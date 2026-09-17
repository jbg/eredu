use super::*;
use crate::working_memory::funding::{RawSpanHostOwner, SpanHostOwner};
use std::sync::{Arc, Barrier};

#[derive(Clone)]
enum LastOwner {
    Full(SpanHostOwner),
    Raw(RawSpanHostOwner),
}

fn detach(
    pool: &WorkingMemoryPool,
    raw: bool,
) -> (
    LastOwner,
    WorkingMemoryReservation,
    WorkingMemoryFundingRun,
    u64,
) {
    let (reservation, run, quote) = accepted(pool);
    let held = quote.span_workspace().retention_peak_bytes().unwrap();
    let (owner, witness) = quote
        .into_funded_span_workspace(&run, &reservation)
        .unwrap();
    assert!(witness.is_none());
    let full = owner.workspace().plan().original_host().unwrap().clone();
    let last = if raw {
        LastOwner::Raw(full.raw().clone())
    } else {
        LastOwner::Full(full.clone())
    };
    drop((full, owner));
    assert_eq!(account(pool, &reservation).1, held);
    assert_eq!(account(pool, &reservation).2, 1);
    (last, reservation, run, held)
}

#[test]
fn original_custody_concurrent_full_and_raw_final_owners_retire_the_same_hold_once() {
    for raw in [false, true] {
        for _ in 0..4 {
            let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
            let source = pool.register_storage([(1u32, 64)]).unwrap();
            let (last, reservation, run, held) = detach(&pool, raw);
            drop((reservation, run, source));
            assert_eq!(pool.used_bytes().unwrap(), held);
            let ready = Arc::new(Barrier::new(5));
            let release = Arc::new(Barrier::new(5));
            std::thread::scope(|scope| {
                let workers = (0..4)
                    .map(|_| {
                        let owner = last.clone();
                        let ready = ready.clone();
                        let release = release.clone();
                        scope.spawn(move || {
                            ready.wait();
                            release.wait();
                            // Both variants exercise the concrete closed owner,
                            // with no outer plan, run or reservation retaining it.
                            match owner {
                                LastOwner::Full(owner) => drop(owner),
                                LastOwner::Raw(owner) => drop(owner),
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                drop(last);
                ready.wait();
                assert_eq!(pool.used_bytes().unwrap(), held);
                release.wait();
                for worker in workers {
                    worker.join().unwrap();
                }
            });
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}

#[test]
fn original_custody_unwind_retirement_never_certifies_an_abandoned_native_scope() {
    for raw in [false, true] {
        for completed in [false, true] {
            let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
            let source = pool.register_storage([(1u32, 64)]).unwrap();
            let (last, reservation, run, held) = detach(&pool, raw);
            let native = run.scope().unwrap();
            if completed {
                native.certify().unwrap();
            } else {
                drop(native);
            }
            drop((reservation, run, source));
            assert!(pool.used_bytes().unwrap() >= held);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _owner = last;
                assert!(pool.used_bytes().unwrap() >= held);
                panic!("unwind the actual last host owner");
            }));
            assert!(result.is_err());
            if completed {
                assert_eq!(pool.used_bytes().unwrap(), 0);
            } else {
                assert!(pool.used_bytes().unwrap() > 0, "native quarantine survives");
            }
        }
    }
}

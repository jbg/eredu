use super::*;
use std::sync::{Barrier, atomic::AtomicUsize};

struct Probe(Arc<AtomicUsize>);
impl Drop for Probe {
    fn drop(&mut self) {
        assert!(runtime_lock::can_reclaim_submission_resources());
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn counter() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}

#[test]
fn specific_retirement_preserves_live_aliases_and_does_not_reap_unrelated_owners() {
    let dropped = counter();
    let unrelated_dropped = counter();
    let value = Array::from_slice(&[3_i32, 7], &[2]);
    value.evaluated().unwrap();
    let alias = value.try_clone_handle().unwrap();
    let unrelated = Array::from_slice(&[11_i32], &[1]);
    unrelated.evaluated().unwrap();
    let (owner, mut receiver) =
        PreparedAllocationOwner::try_new_with_retirement(Probe(dropped.clone())).unwrap();
    owner.try_attach(&value).unwrap();
    unrelated
        .retain_allocation_owner(Probe(unrelated_dropped.clone()))
        .unwrap();
    {
        let _lock = runtime_lock::coordinate_entry();
        drop(unrelated);
        drop(value);
        assert!(!receiver.try_reclaim());
    }
    assert!(
        !receiver.try_reclaim(),
        "the escaped native alias remains live"
    );
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    assert_eq!(unrelated_dropped.load(Ordering::SeqCst), 0);
    drop(alias);
    assert!(receiver.try_reclaim());
    assert!(
        !receiver.try_reclaim(),
        "one actual callback is consumed only once"
    );
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(unrelated_dropped.load(Ordering::SeqCst), 0);
    crate::reclaim_allocation_owners();
    assert_eq!(unrelated_dropped.load(Ordering::SeqCst), 1);
}

#[test]
fn abandoned_specific_receivers_queue_only_after_the_native_backing_retires() {
    for abandon_before in [false, true] {
        let dropped = counter();
        let value = Array::from_slice(&[5_i32], &[1]);
        value.evaluated().unwrap();
        let (owner, receiver) =
            PreparedAllocationOwner::try_new_with_retirement(Probe(dropped.clone())).unwrap();
        owner.try_attach(&value).unwrap();
        if abandon_before {
            drop(receiver);
            crate::reclaim_allocation_owners();
            assert_eq!(dropped.load(Ordering::SeqCst), 0);
            drop(value);
        } else {
            drop(value);
            assert_eq!(dropped.load(Ordering::SeqCst), 0);
            drop(receiver);
        }
        assert_eq!(
            dropped.load(Ordering::SeqCst),
            0,
            "receiver Drop never runs the payload"
        );
        crate::reclaim_allocation_owners();
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }
}

// Exercise the exact one-shot Rust callback with its actual allocated node.
// No C attachment is made here: the empty native node is freed first, and the
// test exclusively transfers the Rust node to one simulated native callback.
fn callback_node<T: Send + 'static>(mut owner: PreparedAllocationOwner<T>) -> usize {
    drop(owner.native.take());
    Box::into_raw(owner.node.take().unwrap()).expose_provenance()
}

#[test]
fn specific_callback_races_receiver_drop_and_reclamation_without_losing_or_duplicating_owners() {
    for reclaim in [false, true] {
        for _ in 0..64 {
            let dropped = counter();
            let (owner, mut receiver) =
                PreparedAllocationOwner::try_new_with_retirement(Probe(dropped.clone())).unwrap();
            let node = callback_node(owner);
            let barrier = Arc::new(Barrier::new(2));
            let other = barrier.clone();
            let worker = std::thread::spawn(move || {
                other.wait();
                // SAFETY: exactly the allocated node/type above, transferred
                // once to this callback after the empty native node was freed.
                unsafe { retire::<Probe>(ptr::with_exposed_provenance_mut(node)) };
            });
            barrier.wait();
            if reclaim {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while !receiver.try_reclaim() {
                    assert!(std::time::Instant::now() < deadline);
                    std::thread::yield_now();
                }
                assert!(!receiver.try_reclaim());
                drop(receiver);
            } else {
                drop(receiver);
            }
            worker.join().unwrap();
            crate::reclaim_allocation_owners();
            assert_eq!(dropped.load(Ordering::SeqCst), 1);
        }
    }
}

#[test]
fn specific_reclamation_preserves_recursion_and_unwind_exclusion() {
    struct Panicking(Arc<AtomicUsize>);
    impl Drop for Panicking {
        fn drop(&mut self) {
            assert!(runtime_lock::can_reclaim_submission_resources());
            assert_eq!(
                crate::reclaim_allocation_owners(),
                0,
                "no reentrant global drain"
            );
            self.0.fetch_add(1, Ordering::SeqCst);
            panic!("specific owner panic");
        }
    }
    let dropped = counter();
    let (owner, mut receiver) =
        PreparedAllocationOwner::try_new_with_retirement(Panicking(dropped.clone())).unwrap();
    let node = callback_node(owner);
    unsafe { retire::<Panicking>(ptr::with_exposed_provenance_mut(node)) };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| receiver.try_reclaim()));
    assert!(result.is_err());
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert!(!receiver.try_reclaim());

    struct DuringUnwind<'a>(&'a mut PreparedAllocationRetirement);
    impl Drop for DuringUnwind<'_> {
        fn drop(&mut self) {
            assert!(!self.0.try_reclaim());
        }
    }
    let next_dropped = counter();
    let (owner, mut next) =
        PreparedAllocationOwner::try_new_with_retirement(Probe(next_dropped.clone())).unwrap();
    let node = callback_node(owner);
    unsafe { retire::<Probe>(ptr::with_exposed_provenance_mut(node)) };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = DuringUnwind(&mut next);
        panic!("caller panic");
    }));
    assert!(result.is_err());
    assert_eq!(next_dropped.load(Ordering::SeqCst), 0);
    assert!(next.try_reclaim(), "unwind and recursion guards must reset");
}

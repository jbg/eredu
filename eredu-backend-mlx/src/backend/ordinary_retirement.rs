//! Preallocated, thread-local deferral of owners whose destructors may block.

use std::{
    cell::{Cell, RefCell},
    ops::{Deref, DerefMut},
};

trait Retired {
    fn take_next(&mut self) -> Option<Box<dyn Retired>>;
    fn set_next(&mut self, next: Option<Box<dyn Retired>>);
}

struct Node<T> {
    value: T,
    next: Option<Box<dyn Retired>>,
}

impl<T> Retired for Node<T> {
    fn take_next(&mut self) -> Option<Box<dyn Retired>> {
        self.next.take()
    }
    fn set_next(&mut self, next: Option<Box<dyn Retired>>) {
        self.next = next;
    }
}

#[derive(Default)]
struct Queue(Option<Box<dyn Retired>>);

impl Drop for Queue {
    fn drop(&mut self) {
        // These owners already have terminal-work proof. Thread exit is their
        // final owner-thread boundary; do not abandon them when there is no
        // subsequent API call. Detach links before running arbitrary Drop.
        while let Some(mut node) = self.0.take() {
            self.0 = node.take_next();
            drop(node);
        }
    }
}

thread_local! {
    static RETIRED: RefCell<Queue> = RefCell::default();
    static RECLAIMING: Cell<bool> = const { Cell::new(false) };
    static TRIM_CACHE: Cell<bool> = const { Cell::new(false) };
}

// Initialize terminal queues before recovery TLS. Rust destroys TLS in reverse
// initialization order, so recovery can still stage terminal owners at exit.
pub(crate) fn prepare_thread() {
    let _ = RETIRED.try_with(|_| {});
}

pub(crate) fn thread_available() -> bool {
    RETIRED.try_with(|_| ()).is_ok()
}

/// Last field of a public execution owner. Runs after its payload references
/// have been released, outside native recovery and ordinary host locks.
pub(crate) struct ReclaimOnDrop;

impl Drop for ReclaimOnDrop {
    fn drop(&mut self) {
        if safemlx::can_reclaim_submission_resources()
            && !RECLAIMING.try_with(Cell::get).unwrap_or(true)
        {
            crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
            reclaim_all();
        }
    }
}

/// Last field of a retired executable owner: schedules allocator cleanup after
/// its resources finish ordinary-host destruction, never inside native Drop.
pub(crate) struct TrimCacheOnRetirement;

impl Drop for TrimCacheOnRetirement {
    fn drop(&mut self) {
        let _ = TRIM_CACHE.try_with(|pending| pending.set(true));
    }
}

/// Preallocated ownership staged for explicit, unlocked ordinary-host cleanup.
pub(crate) struct OrdinaryRetirement<T: 'static>(Option<Box<Node<T>>>);

impl<T: 'static> OrdinaryRetirement<T> {
    pub(crate) fn new(value: T) -> Self {
        prepare_thread();
        Self(Some(Box::new(Node { value, next: None })))
    }

    /// Transfers an unretired owner to an ordinary caller without running Drop.
    pub(crate) fn into_inner(mut self) -> T {
        self.0.take().expect("live retirement owner").value
    }
}

impl<T: 'static> Deref for OrdinaryRetirement<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0.as_ref().expect("live retirement owner").value
    }
}

impl<T: 'static> DerefMut for OrdinaryRetirement<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0.as_mut().expect("live retirement owner").value
    }
}

impl<T: 'static> Drop for OrdinaryRetirement<T> {
    fn drop(&mut self) {
        let Some(owned) = self.0.take() else {
            return;
        };
        let mut node = Some(owned as Box<dyn Retired>);
        let _ = RETIRED.try_with(|retired| {
            if let Ok(mut queue) = retired.try_borrow_mut() {
                let mut owned = node.take().expect("one preallocated retirement node");
                owned.set_next(queue.0.take());
                queue.0 = Some(owned);
            }
        });
        // TLS can be unavailable while its final queue is being destroyed.
        // The node is already safe to release; normal Rust ownership still
        // applies to recursively retired owners at that boundary.
        drop(node);
    }
}

/// Reclaims only an already-retired snapshot, outside native/list locks.
/// Recursive/new work waits for a later call; snapshots retain ordinary Rust Drop semantics.
pub(crate) fn reclaim() {
    if !safemlx::can_reclaim_submission_resources() {
        return;
    }
    let _ = RECLAIMING.try_with(|reclaiming| {
        if reclaiming.replace(true) {
            return;
        }
        struct Reset<'a>(&'a Cell<bool>);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.set(false);
            }
        }
        let _reset = Reset(reclaiming);
        let pending = RETIRED
            .try_with(|retired| {
                retired
                    .try_borrow_mut()
                    .ok()
                    .and_then(|mut queue| queue.0.take())
            })
            .ok()
            .flatten();
        let mut pending = Queue(pending);
        while let Some(mut node) = pending.0.take() {
            pending.0 = node.take_next();
            drop(node);
        }
        // No native query on ordinary frame retirement. A final executable
        // requests cleanup only after nested owners have finished destruction.
        if TRIM_CACHE.try_with(Cell::get).unwrap_or(false) {
            let empty = RETIRED
                .try_with(|retired| retired.borrow().0.is_none())
                .unwrap_or(false);
            if empty && safemlx::memory::clear_cache().is_ok() {
                let _ = TRIM_CACHE.try_with(|pending| pending.set(false));
            }
        }
    });
}

/// Finishes staged destruction at an explicit synchronous host boundary.
/// Destructors may stage further owners, so consume those later snapshots too.
pub(crate) fn reclaim_all() {
    if !safemlx::can_reclaim_submission_resources()
        || RECLAIMING.try_with(Cell::get).unwrap_or(true)
    {
        return;
    }
    loop {
        reclaim();
        if !RETIRED
            .try_with(|retired| retired.borrow().0.is_some())
            .unwrap_or(false)
        {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct Witness(Arc<AtomicUsize>);
    impl Drop for Witness {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn ordinary_retirement_into_inner_transfers_identity_without_retirement() {
        let drops = Arc::new(AtomicUsize::new(0));
        let owner = OrdinaryRetirement::new(Witness(Arc::clone(&drops)));
        let witness = owner.into_inner();
        assert!(Arc::ptr_eq(&drops, &witness.0));
        reclaim();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(witness);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn ordinary_retirement_never_drops_owners_under_the_runtime_lock() {
        let drops = Arc::new(AtomicUsize::new(0));
        let mut owner = Some(OrdinaryRetirement::new(Witness(Arc::clone(&drops))));
        loop {
            if safemlx::try_with_submission_retirement(|| {
                drop(owner.take());
                reclaim();
                assert_eq!(drops.load(Ordering::SeqCst), 0);
            })
            .is_some()
            {
                break;
            }
            std::thread::yield_now();
        }
        reclaim();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn ordinary_retirement_defers_reentrant_new_owners_to_the_next_snapshot() {
        struct Reentrant(Arc<AtomicUsize>);
        impl Drop for Reentrant {
            fn drop(&mut self) {
                drop(OrdinaryRetirement::new(Witness(Arc::clone(&self.0))));
                reclaim();
                assert_eq!(self.0.load(Ordering::SeqCst), 0);
            }
        }
        let drops = Arc::new(AtomicUsize::new(0));
        drop(OrdinaryRetirement::new(Reentrant(Arc::clone(&drops))));
        reclaim();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        reclaim();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

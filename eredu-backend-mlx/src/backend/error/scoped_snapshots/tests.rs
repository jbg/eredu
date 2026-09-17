use super::*;
use eredu_core::BackendFailureKind;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Debug)]
struct Cause {
    id: u8,
    drops: Arc<AtomicUsize>,
}
impl std::fmt::Display for Cause {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        panic!("snapshot formatted")
    }
}
impl std::error::Error for Cause {}
impl Drop for Cause {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
fn same(a: &Cause, b: &Cause) -> bool {
    a.id == b.id
}

#[test]
fn cached_source_reuse_keeps_the_incoming_owner_until_caller_retires_it() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut cache = SnapshotCache::<2>::new();
    let source = SharedBackendFailure::new(
        BackendFailureKind::Busy,
        Cause {
            id: 7,
            drops: drops.clone(),
        },
    );
    let first = cache.publish(0, source);
    let incoming = Cause {
        id: 7,
        drops: drops.clone(),
    };
    let SnapshotLookup::Retained(second) = cache.lookup(Some(0), &incoming, same) else {
        panic!("source not reused")
    };
    assert!(std::ptr::eq(
        first.source_error().downcast_ref::<Cause>().unwrap(),
        second.source_error().downcast_ref::<Cause>().unwrap()
    ));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(incoming);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(cache);
    drop(first);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(second);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn first_terminal_source_survives_all_later_keys_without_comparison_or_replacement() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut cache = SnapshotCache::<2>::new();
    let first = cache.publish(
        1,
        SharedBackendFailure::new(
            BackendFailureKind::Other,
            Cause {
                id: 1,
                drops: drops.clone(),
            },
        ),
    );
    let changed = Cause {
        id: 2,
        drops: drops.clone(),
    };
    assert!(matches!(
        cache.lookup(Some(1), &changed, same),
        SnapshotLookup::Conflict(SnapshotConflict::Changed(1))
    ));
    let terminal = cache.publish_terminal(SharedBackendFailure::new(
        BackendFailureKind::InvalidSession,
        changed,
    ));
    let incoming = Cause {
        id: 3,
        drops: drops.clone(),
    };
    let SnapshotLookup::Retained(later) =
        cache.lookup(None, &incoming, |_, _| panic!("terminal compared"))
    else {
        panic!("terminal lost")
    };
    assert!(std::ptr::eq(
        terminal.source_error().downcast_ref::<Cause>().unwrap(),
        later.source_error().downcast_ref::<Cause>().unwrap()
    ));
    drop(incoming);
    drop(cache);
    drop(terminal);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(first);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    drop(later);
    assert_eq!(drops.load(Ordering::SeqCst), 3);
}

#[test]
fn unknown_and_out_of_range_keys_do_not_publish_or_consume_the_input() {
    let drops = Arc::new(AtomicUsize::new(0));
    let cache = SnapshotCache::<1>::new();
    let incoming = Cause {
        id: 1,
        drops: drops.clone(),
    };
    for key in [None, Some(1), Some(usize::MAX)] {
        assert!(matches!(
            cache.lookup(key, &incoming, same),
            SnapshotLookup::Conflict(SnapshotConflict::Unclassified)
        ));
    }
    assert!(matches!(
        cache.lookup(Some(0), &incoming, same),
        SnapshotLookup::Vacant(0)
    ));
    assert!(cache.terminal().is_none());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(incoming);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

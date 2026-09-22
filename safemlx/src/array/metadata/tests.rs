use super::*;
use crate::{ops::indexing::TryIndexOp, Device, DeviceType, Stream};
use std::{
    cell::Cell,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
    time::Duration,
};

thread_local! { static HOUSEKEEPING_CALLS: Cell<usize> = const { Cell::new(0) }; }
fn record_housekeeping() {
    HOUSEKEEPING_CALLS.with(|calls| calls.set(calls.get() + 1));
}
struct Hook;
impl Drop for Hook {
    fn drop(&mut self) {
        runtime_lock::unregister_housekeeping_hook(record_housekeeping);
    }
}

#[test]
fn snapshots_keep_lazy_storage_unknown_and_completed_alias_capacity_without_housekeeping() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let root = Array::from_slice(&[1_i32, 2, 3, 4, 5, 6], &[2, 3]);
    let allocation = root.allocation_info().unwrap().unwrap();
    let tail = root.try_index_device((.., 2..), &stream).unwrap();
    runtime_lock::register_housekeeping_hook(record_housekeeping);
    let _hook = Hook;
    HOUSEKEEPING_CALLS.with(|calls| calls.set(0));
    let snapshot = root.try_metadata_snapshot().unwrap();
    assert_eq!(snapshot.shape(), [2, 3]);
    assert_eq!(snapshot.dtype(), Dtype::Int32);
    assert_eq!(snapshot.nbytes(), 24);
    assert_eq!(snapshot.allocation(), Some(allocation));
    let lazy = tail.try_metadata_snapshot().unwrap();
    assert_eq!(lazy.shape(), [2, 1]);
    assert_eq!(lazy.nbytes(), 8);
    assert_eq!(lazy.allocation(), None);
    assert_eq!(HOUSEKEEPING_CALLS.with(Cell::get), 0);
    let evaluated = tail.evaluated().unwrap();
    HOUSEKEEPING_CALLS.with(|calls| calls.set(0));
    let completed = tail.try_metadata_snapshot().unwrap();
    assert_eq!(completed.allocation(), Some(allocation));
    assert!(allocation.bytes() > completed.nbytes());
    assert_eq!(HOUSEKEEPING_CALLS.with(Cell::get), 0);
    let values = evaluated.deep_clone().unwrap();
    assert_eq!(values.as_slice::<i32>(), [3, 6]);
}

#[test]
fn snapshot_returns_busy_without_waiting_for_a_foreign_runtime_lock() {
    let array = Array::from_slice(&[1_i32, -2, 3], &[3]);
    let (locked_tx, locked_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        locked_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    });
    locked_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let result = array.try_metadata_snapshot();
    release_tx.send(()).unwrap();
    worker.join().unwrap();
    assert!(matches!(result, Err(ArrayMetadataError::RuntimeBusy)));
    assert_eq!(array.try_metadata_snapshot().unwrap().nbytes(), 12);
}

struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn snapshot_neither_reclaims_queued_owners_nor_retains_observed_backing() {
    let observed = Array::from_slice(&[3_i32, 5, 7], &[3]);
    let retired = Arc::new(AtomicUsize::new(0));
    let queued = Array::from_slice(&[11_i32, 13], &[2]);
    queued
        .retain_allocation_owner(Retired(Arc::clone(&retired)))
        .unwrap();
    runtime_lock::try_retire(|| drop(queued)).unwrap();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    let snapshot = observed.try_metadata_snapshot().unwrap();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    // The physical root remains owned by the allocator cache after its last
    // Array disappears. Only explicit cache eviction queues that custody.
    crate::memory::clear_cache().unwrap();
    crate::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    observed
        .retain_allocation_owner(Retired(Arc::clone(&retired)))
        .unwrap();
    drop(observed);
    crate::memory::clear_cache().unwrap();
    crate::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 2);
    assert!(snapshot.allocation().is_some());
}

#[test]
fn inspection_clones_share_known_and_lazy_descriptors_without_housekeeping() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let root = Array::from_slice(&[2_i32, 3, 5, 7, 11, 13], &[2, 3]);
    let retired = Arc::new(AtomicUsize::new(0));
    root.retain_allocation_owner(Retired(Arc::clone(&retired)))
        .unwrap();
    let view = root.try_index_device((.., 2..), &stream).unwrap();
    view.evaluated().unwrap();
    let lazy = root.square(&stream).unwrap();
    let root_info = root.try_metadata_snapshot().unwrap().allocation().unwrap();
    assert!(root_info.bytes() > view.nbytes());
    assert!(lazy.try_metadata_snapshot().unwrap().allocation().is_none());
    runtime_lock::register_housekeeping_hook(record_housekeeping);
    let hook = Hook;
    HOUSEKEEPING_CALLS.with(|calls| calls.set(0));
    let root_copy = root.try_clone_for_inspection().unwrap();
    let view_copy = view.try_clone_for_inspection().unwrap();
    let lazy_copy = lazy.try_clone_for_inspection().unwrap();
    assert_eq!(
        root_copy.try_metadata_snapshot().unwrap().allocation(),
        Some(root_info)
    );
    assert_eq!(
        view_copy.try_metadata_snapshot().unwrap().allocation(),
        Some(root_info)
    );
    assert!(lazy_copy
        .try_metadata_snapshot()
        .unwrap()
        .allocation()
        .is_none());
    assert!(lazy.try_metadata_snapshot().unwrap().allocation().is_none());
    assert_eq!(HOUSEKEEPING_CALLS.with(Cell::get), 0);
    drop(hook);
    drop((root, view, lazy));
    crate::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(
        root_copy.evaluated().unwrap().as_slice::<i32>(),
        [2, 3, 5, 7, 11, 13]
    );
    assert_eq!(
        view_copy.evaluated().unwrap().try_to_vec::<i32>().unwrap(),
        [5, 13]
    );
    assert_eq!(
        lazy_copy.evaluated().unwrap().as_slice::<i32>(),
        [4, 9, 25, 49, 121, 169]
    );
    drop((lazy_copy, root_copy));
    crate::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(view_copy);
    crate::memory::clear_cache().unwrap();
    crate::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn inspection_clone_returns_busy_before_foreign_runtime_owner_is_released() {
    let root = Array::from_slice(&[17_i32, 19], &[2]);
    let original = root.try_metadata_snapshot().unwrap();
    let (locked_tx, locked_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        locked_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    });
    locked_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let cloned = root.try_clone_for_inspection();
    release_tx.send(()).unwrap();
    worker.join().unwrap();
    assert!(matches!(cloned, Err(ArrayMetadataError::RuntimeBusy)));
    assert_eq!(root.try_metadata_snapshot().unwrap(), original);
    let copy = root.try_clone_for_inspection().unwrap();
    assert_eq!(copy.try_metadata_snapshot().unwrap(), original);
}

#[test]
fn inspection_clone_does_not_reclaim_unrelated_queued_owners() {
    let root = Array::from_slice(&[23_i32, 29], &[2]);
    let queued = Array::from_slice(&[31_i32, 37], &[2]);
    let retired = Arc::new(AtomicUsize::new(0));
    queued
        .retain_allocation_owner(Retired(Arc::clone(&retired)))
        .unwrap();
    runtime_lock::try_retire(|| drop(queued)).unwrap();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    runtime_lock::register_housekeeping_hook(record_housekeeping);
    let hook = Hook;
    HOUSEKEEPING_CALLS.with(|calls| calls.set(0));
    let copy = root.try_clone_for_inspection().unwrap();
    assert_eq!(HOUSEKEEPING_CALLS.with(Cell::get), 0);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    drop(hook);
    crate::memory::clear_cache().unwrap();
    crate::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    drop(root);
    assert_eq!(copy.evaluated().unwrap().as_slice::<i32>(), [23, 29]);
}

#[test]
fn allocation_only_inspection_keeps_lazy_unknown_and_full_alias_capacity() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let root = Array::from_slice(&[2_i32, -3, 5, 7, 11, -13], &[2, 3]);
    let expected = root.allocation_info().unwrap().unwrap();
    let tail = root.try_index_device((.., 2..), &stream).unwrap();
    let lazy = root.square(&stream).unwrap();
    runtime_lock::register_housekeeping_hook(record_housekeeping);
    let hook = Hook;
    HOUSEKEEPING_CALLS.with(|calls| calls.set(0));
    assert_eq!(root.try_allocation_info().unwrap(), Some(expected));
    assert_eq!(tail.try_allocation_info().unwrap(), None);
    assert_eq!(lazy.try_allocation_info().unwrap(), None);
    assert_eq!(HOUSEKEEPING_CALLS.with(Cell::get), 0);
    drop(hook);
    let evaluated = tail.evaluated().unwrap();
    runtime_lock::register_housekeeping_hook(record_housekeeping);
    let hook = Hook;
    HOUSEKEEPING_CALLS.with(|calls| calls.set(0));
    assert_eq!(tail.try_allocation_info().unwrap(), Some(expected));
    assert!(expected.bytes() > tail.nbytes());
    assert_eq!(lazy.try_allocation_info().unwrap(), None);
    assert_eq!(HOUSEKEEPING_CALLS.with(Cell::get), 0);
    drop(hook);
    assert_eq!(evaluated.try_to_vec::<i32>().unwrap(), [5, -13]);
}

#[test]
fn allocation_only_inspection_returns_busy_before_foreign_runtime_release() {
    let root = Array::from_slice(&[17_i32, -19], &[2]);
    let expected = root.allocation_info().unwrap();
    let (locked_tx, locked_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        locked_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    });
    locked_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let result = root.try_allocation_info();
    release_tx.send(()).unwrap();
    worker.join().unwrap();
    assert!(matches!(result, Err(ArrayMetadataError::RuntimeBusy)));
    assert_eq!(root.try_allocation_info().unwrap(), expected);
    assert_eq!(root.evaluated().unwrap().as_slice::<i32>(), [17, -19]);
}

#[test]
fn allocation_only_inspection_neither_reclaims_nor_pins_native_owners() {
    let observed = Array::from_slice(&[23_i32, 29], &[2]);
    let queued = Array::from_slice(&[31_i32, 37], &[2]);
    let retired = Arc::new(AtomicUsize::new(0));
    observed
        .retain_allocation_owner(Retired(Arc::clone(&retired)))
        .unwrap();
    queued
        .retain_allocation_owner(Retired(Arc::clone(&retired)))
        .unwrap();
    runtime_lock::try_retire(|| drop(queued)).unwrap();
    runtime_lock::register_housekeeping_hook(record_housekeeping);
    let hook = Hook;
    HOUSEKEEPING_CALLS.with(|calls| calls.set(0));
    let facts = observed.try_allocation_info().unwrap().unwrap();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(HOUSEKEEPING_CALLS.with(Cell::get), 0);
    drop(hook);
    crate::memory::clear_cache().unwrap();
    crate::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    drop(observed);
    crate::memory::clear_cache().unwrap();
    crate::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 2);
    assert!(facts.bytes() >= 2 * std::mem::size_of::<i32>());
}

#[test]
fn inspection_handle_size_is_cold_while_runtime_is_owned_and_retirement_is_pending() {
    let retired = Arc::new(AtomicUsize::new(0));
    let queued = Array::from_slice(&[41_i32, -43], &[2]);
    queued
        .retain_allocation_owner(Retired(Arc::clone(&retired)))
        .unwrap();
    runtime_lock::try_retire(|| drop(queued)).unwrap();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    runtime_lock::register_housekeeping_hook(record_housekeeping);
    let hook = Hook;
    HOUSEKEEPING_CALLS.with(|calls| calls.set(0));
    let before = Array::inspection_clone_handle_bytes();
    let (locked_tx, locked_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        // Acquire the foreign runtime lock without draining queued owners.
        runtime_lock::try_retire(|| {
            locked_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        })
        .unwrap();
    });
    locked_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let while_owned = Array::inspection_clone_handle_bytes();
    let housekeeping = HOUSEKEEPING_CALLS.with(Cell::get);
    let reclaimed = retired.load(Ordering::SeqCst);
    release_tx.send(()).unwrap();
    worker.join().unwrap();
    assert!(before > 0);
    assert_eq!(while_owned, before);
    assert_eq!(housekeeping, 0);
    assert_eq!(reclaimed, 0);
    drop(hook);
    crate::memory::clear_cache().unwrap();
    crate::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn ordinary_inventory_keeps_one_runtime_owner_and_does_not_complete_lazy_arrays() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let root = Array::from_slice(&[2_i32, -3, 5, 7], &[2, 2]);
    let expected = root.allocation_info().unwrap();
    let lazy = root.try_index_device((.., 1..), &stream).unwrap();
    runtime_lock::register_housekeeping_hook(record_housekeeping);
    let hook = Hook;
    HOUSEKEEPING_CALLS.with(|calls| calls.set(0));
    let inspection = OrdinaryArrayMetadataGuard::enter().unwrap();
    assert_eq!(HOUSEKEEPING_CALLS.with(Cell::get), 1);
    let clone = root.try_clone_for_inspection().unwrap();
    assert_eq!(root.try_allocation_info().unwrap(), expected);
    assert_eq!(clone.try_allocation_info().unwrap(), expected);
    assert_eq!(lazy.try_allocation_info().unwrap(), None);
    assert_eq!(HOUSEKEEPING_CALLS.with(Cell::get), 1);
    // A different thread remains a cold immediate caller, even though this
    // thread explicitly chose ordinary serialization for the whole inventory.
    let foreign = root.clone();
    let busy = std::thread::spawn(move || {
        let result = foreign.try_allocation_info();
        (foreign, result)
    })
    .join()
    .unwrap();
    assert!(matches!(busy.1, Err(ArrayMetadataError::RuntimeBusy)));
    drop(inspection);
    drop(hook);
    let foreign = busy.0;
    assert_eq!(
        std::thread::spawn(move || foreign.try_allocation_info())
            .join()
            .unwrap()
            .unwrap(),
        expected
    );
    assert_eq!(clone.evaluated().unwrap().as_slice::<i32>(), [2, -3, 5, 7]);
    assert_eq!(lazy.try_allocation_info().unwrap(), None);
}

#[test]
fn ordinary_inventory_waits_before_inspection_without_repeating_a_query() {
    let root = Array::from_slice(&[11_i32, -13], &[2]);
    let expected = root.allocation_info().unwrap();
    let owner = runtime_lock::enter();
    let (started_tx, started_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        let facts = {
            let _inspection = OrdinaryArrayMetadataGuard::enter().unwrap();
            root.try_allocation_info()
        };
        finished_tx.send(facts).unwrap();
    });
    started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let blocked = matches!(
        finished_rx.recv_timeout(Duration::from_millis(20)),
        Err(mpsc::RecvTimeoutError::Timeout)
    );
    drop(owner);
    let actual = finished_rx
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap();
    worker.join().unwrap();
    assert!(
        blocked,
        "ordinary entry cannot run the query under a foreign owner"
    );
    assert_eq!(actual, expected);
}

#[test]
fn ordinary_inventory_refuses_active_original_before_waiting_or_housekeeping() {
    let mut scope = crate::SubmissionScope::begin().unwrap();
    scope.require_original_native_controls().unwrap();
    let status = scope.status();
    runtime_lock::register_housekeeping_hook(record_housekeeping);
    let hook = Hook;
    HOUSEKEEPING_CALLS.with(|calls| calls.set(0));
    let (held_tx, held_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _owner = runtime_lock::enter();
        held_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
    });
    held_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let result = OrdinaryArrayMetadataGuard::enter();
    let released = release_tx.send(()).is_ok();
    let held = worker.join().unwrap();
    assert!(
        released && held,
        "original refusal must precede lock waiting"
    );
    assert!(matches!(
        result,
        Err(crate::OriginalNativeControlError::ForeignDomain)
    ));
    assert_eq!(scope.status(), status);
    assert_eq!(HOUSEKEEPING_CALLS.with(Cell::get), 0);
    drop(hook);
    scope.seal();
}

use std::{cell::Cell, sync::mpsc, time::Duration};

use super::*;

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

fn buffer() -> ImmutableHostTransferBuffer {
    let mut buffer =
        HostTransferBuffer::new(&[2, 3], Dtype::Int32, HostTransferPolicy::Transfer).unwrap();
    let bytes = [1_i32, -2, 3, 4, 5, 6]
        .into_iter()
        .flat_map(i32::to_ne_bytes)
        .collect::<Vec<_>>();
    buffer.as_bytes_mut().unwrap().copy_from_slice(&bytes);
    buffer.freeze()
}

#[test]
fn immutable_metadata_snapshot_preserves_facts_without_housekeeping() {
    let buffer = buffer();
    let allocation = buffer.allocation_info().unwrap();
    HOUSEKEEPING_CALLS.with(|calls| calls.set(0));
    runtime_lock::register_housekeeping_hook(record_housekeeping);
    let _hook = Hook;
    let snapshot = buffer.try_metadata_snapshot().unwrap();
    assert_eq!(snapshot.shape(), [2, 3]);
    assert_eq!(snapshot.dtype(), Dtype::Int32);
    assert!(snapshot.copy_output_layout().row_contiguous());
    assert_eq!(snapshot.nbytes(), 24);
    assert_eq!(snapshot.allocation(), allocation);
    assert_eq!(snapshot.policy(), HostTransferPolicy::Transfer);
    assert_eq!(HOUSEKEEPING_CALLS.with(Cell::get), 0);
    assert_eq!(buffer.try_metadata_snapshot().unwrap(), snapshot);
    assert_eq!(HOUSEKEEPING_CALLS.with(Cell::get), 0);
    assert_eq!(buffer.nbytes().unwrap(), 24);
    assert_eq!(
        HOUSEKEEPING_CALLS.with(Cell::get),
        1,
        "ordinary getters keep their existing housekeeping behavior"
    );
}

#[test]
fn immutable_metadata_snapshot_returns_busy_without_waiting_for_foreign_runtime() {
    let buffer = buffer();
    let (locked_tx, locked_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        locked_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    });
    locked_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    // The worker still owns the lock until after this call returns.
    let result = buffer.try_metadata_snapshot();
    release_tx.send(()).unwrap();
    worker.join().unwrap();
    assert!(matches!(
        result,
        Err(HostTransferMetadataError::RuntimeBusy)
    ));
    assert_eq!(buffer.try_metadata_snapshot().unwrap().nbytes(), 24);
}

#[test]
fn allocation_only_host_inspection_preserves_capacity_without_housekeeping_or_pin() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    struct Retired(Arc<AtomicUsize>);
    impl Drop for Retired {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let observed = buffer();
    let queued = buffer();
    let expected = observed.allocation_info().unwrap();
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
    let facts = observed.try_allocation_info().unwrap();
    assert_eq!(facts, expected);
    assert!(facts.bytes() >= 24);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(HOUSEKEEPING_CALLS.with(Cell::get), 0);
    drop(hook);
    let expected_bytes = [1_i32, -2, 3, 4, 5, 6]
        .into_iter()
        .flat_map(i32::to_ne_bytes)
        .collect::<Vec<_>>();
    assert_eq!(observed.as_bytes().unwrap(), expected_bytes);
    crate::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    drop(observed);
    crate::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 2);
    assert_eq!(facts, expected);
}

#[test]
fn allocation_only_host_inspection_returns_busy_before_foreign_runtime_release() {
    let observed = buffer();
    let expected = observed.allocation_info().unwrap();
    let (locked_tx, locked_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        locked_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    });
    locked_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let result = observed.try_allocation_info();
    release_tx.send(()).unwrap();
    worker.join().unwrap();
    assert!(matches!(
        result,
        Err(HostTransferMetadataError::RuntimeBusy)
    ));
    assert_eq!(observed.try_allocation_info().unwrap(), expected);
}

#[test]
fn fixed_host_descriptor_matches_actual_metadata_and_does_not_retain_source() {
    let source = std::sync::Arc::new(buffer());
    let weak = std::sync::Arc::downgrade(&source);
    let ordinary = source.try_metadata_snapshot().unwrap();
    HOUSEKEEPING_CALLS.with(|calls| calls.set(0));
    runtime_lock::register_housekeeping_hook(record_housekeeping);
    let hook = Hook;
    let fixed = source.try_fixed_descriptor::<4>().unwrap();
    assert_eq!(fixed.shape(), ordinary.shape());
    assert_eq!(fixed.dtype(), ordinary.dtype());
    assert_eq!(fixed.nbytes(), ordinary.nbytes());
    assert_eq!(fixed.allocation(), ordinary.allocation());
    assert_eq!(fixed.policy(), ordinary.policy());
    assert_eq!(fixed.storage_kind(), ordinary.storage_kind());
    assert!(matches!(
        source.try_fixed_descriptor::<1>(),
        Err(HostTransferMetadataError::ShapeCapacity)
    ));
    assert_eq!(HOUSEKEEPING_CALLS.with(Cell::get), 0);
    drop(hook);
    drop(source);
    assert!(weak.upgrade().is_none());
    assert_eq!(fixed.shape(), [2, 3]);
}

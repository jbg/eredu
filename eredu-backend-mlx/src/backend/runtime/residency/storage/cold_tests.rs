use super::*;
use safemlx::{
    ops::indexing::TryIndexOp, ArrayMetadataError, Device, DeviceType, HostTransferBuffer,
    HostTransferMetadataError, HostTransferPolicy, Stream,
};
use std::{
    cell::Cell,
    sync::atomic::{AtomicUsize, Ordering},
};

thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.with(|count| count.set(count.get() + 1));
}
struct Hook;
impl Hook {
    fn install() -> Self {
        safemlx::register_thread_runtime_housekeeping(housekeeping);
        HOUSEKEEPING.with(|count| count.set(0));
        Self
    }
}
impl Drop for Hook {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}
fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}
fn facts(
    storage: &RetainedStorage,
) -> (
    Option<u64>,
    BTreeMap<safemlx::AllocationIdentity, u64>,
    usize,
) {
    (
        storage.byte_bound().unwrap(),
        storage.array_allocation_facts(),
        storage.unknown_arrays().len(),
    )
}
fn host_metadata_error(error: &ResidencyError) -> Option<&HostTransferMetadataError> {
    let mut current: &(dyn std::error::Error + 'static) = error;
    loop {
        if let Some(error) = current.downcast_ref::<HostTransferMetadataError>() {
            return Some(error);
        }
        current = current.source()?;
    }
}
fn metadata_error(error: &ResidencyError) -> Option<&ArrayMetadataError> {
    let mut current: &(dyn std::error::Error + 'static) = error;
    loop {
        if let Some(error) = current.downcast_ref::<ArrayMetadataError>() {
            return Some(error);
        }
        current = current.source()?;
    }
}

#[test]
fn retained_array_inspection_and_handle_retention_never_run_housekeeping() {
    let stream = stream();
    let root = Array::from_slice(&[3_i32, -5, 7, 11, 13, 17], &[2, 3]);
    let view = root.try_index_device((.., 2..), &stream).unwrap();
    view.evaluated().unwrap();
    let lazy = root.square(&stream).unwrap();
    let allocation = root.try_metadata_snapshot().unwrap().allocation().unwrap();
    assert_eq!(
        view.try_metadata_snapshot().unwrap().allocation(),
        Some(allocation)
    );
    assert!(allocation.bytes() > view.try_metadata_snapshot().unwrap().nbytes());
    assert!(lazy.try_metadata_snapshot().unwrap().allocation().is_none());
    let hook = Hook::install();
    let mut storage = RetainedStorage::default();
    storage.include_array(&root).unwrap();
    storage.include_array(&view).unwrap();
    storage.include_array(&root).unwrap();
    assert_eq!(
        storage.byte_bound().unwrap(),
        Some(allocation.bytes() as u64)
    );
    assert_eq!(
        storage.array_allocation_facts(),
        BTreeMap::from([(allocation.identity(), allocation.bytes() as u64)])
    );
    storage.include_array(&lazy).unwrap();
    assert_eq!(storage.unknown_arrays().len(), 1);
    assert_eq!(storage.byte_bound().unwrap(), None);
    assert!(lazy.try_metadata_snapshot().unwrap().allocation().is_none());
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    let captured = facts(&storage);
    drop(hook);
    // Explicit execution is outside inventory. It does not retroactively
    // replace the unknown facts already captured by the earlier inventory.
    assert_eq!(
        lazy.evaluated().unwrap().as_slice::<i32>(),
        [9, 25, 49, 121, 169, 289]
    );
    assert_eq!(facts(&storage), captured);
    let completed = lazy.try_metadata_snapshot().unwrap().allocation().unwrap();
    let hook = Hook::install();
    let mut known = RetainedStorage::default();
    known.include_array(&lazy).unwrap();
    assert_eq!(
        known.array_allocation_facts(),
        BTreeMap::from([(completed.identity(), completed.bytes() as u64)])
    );
    assert_eq!(known.byte_bound().unwrap(), Some(completed.bytes() as u64));
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    drop(hook);
    drop((root, view, lazy));
    assert_eq!(
        known
            .arrays
            .values()
            .next()
            .unwrap()
            .owned()
            .unwrap()
            .1
            .evaluated()
            .unwrap()
            .as_slice::<i32>(),
        [9, 25, 49, 121, 169, 289]
    );
}

#[test]
fn inspection_errors_keep_typed_busy_and_original_native_causes() {
    let busy = array_inspection_error(ArrayMetadataError::RuntimeBusy);
    assert!(matches!(
        metadata_error(&busy),
        Some(ArrayMetadataError::RuntimeBusy)
    ));
    #[derive(Debug, thiserror::Error)]
    #[error("native descriptor inspection failed")]
    struct NativeFailure;
    let error = array_inspection_error(ArrayMetadataError::Native(
        safemlx::error::Exception::from_source(NativeFailure),
    ));
    let Some(ArrayMetadataError::Native(native)) = metadata_error(&error) else {
        panic!("native metadata cause must remain typed");
    };
    assert!(std::error::Error::source(native)
        .unwrap()
        .downcast_ref::<NativeFailure>()
        .is_some());
    let busy = host_inspection_error(HostTransferMetadataError::RuntimeBusy);
    assert!(matches!(
        host_metadata_error(&busy),
        Some(HostTransferMetadataError::RuntimeBusy)
    ));
    let error = host_inspection_error(HostTransferMetadataError::Native(
        safemlx::error::Exception::from_source(NativeFailure),
    ));
    let Some(HostTransferMetadataError::Native(native)) = host_metadata_error(&error) else {
        panic!("host metadata cause must remain typed");
    };
    assert!(std::error::Error::source(native)
        .unwrap()
        .downcast_ref::<NativeFailure>()
        .is_some());
}

struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
#[test]
fn retained_array_inspection_does_not_reclaim_unrelated_queued_owners() {
    let root = Array::from_slice(&[23_i32, 29], &[2]);
    let queued = Array::from_slice(&[31_i32, 37], &[2]);
    queued.evaluated().unwrap();
    let retired = Arc::new(AtomicUsize::new(0));
    queued
        .retain_allocation_owner(Retired(Arc::clone(&retired)))
        .unwrap();
    safemlx::try_with_submission_retirement(|| drop(queued)).unwrap();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    let hook = Hook::install();
    let mut storage = RetainedStorage::default();
    storage.include_array(&root).unwrap();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    drop(hook);
    safemlx::memory::clear_cache().unwrap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    assert!(storage.byte_bound().unwrap().unwrap() > 0);
}

fn host_buffer() -> Arc<ImmutableHostTransferBuffer> {
    let mut buffer =
        HostTransferBuffer::new(&[2, 3], safemlx::Dtype::Int32, HostTransferPolicy::Transfer)
            .unwrap();
    buffer.as_bytes_mut().unwrap().copy_from_slice(
        &[3_i32, -5, 7, 11, 13, 17]
            .into_iter()
            .flat_map(i32::to_ne_bytes)
            .collect::<Vec<_>>(),
    );
    Arc::new(buffer.freeze())
}

#[test]
fn retained_host_inspection_duplicates_array_aliases_and_drop_are_cold() {
    let devices = if cfg!(feature = "metal") {
        vec![DeviceType::Cpu, DeviceType::Gpu]
    } else {
        vec![DeviceType::Cpu]
    };
    for device in devices {
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let host = host_buffer();
        let allocation = host.try_metadata_snapshot().unwrap().allocation();
        let array = host.copy_to_array(&stream).unwrap().synchronize().unwrap();
        array.evaluated().unwrap();
        let view = array.try_index_device((.., 2..), &stream).unwrap();
        view.evaluated().unwrap();
        let native = view.try_metadata_snapshot().unwrap().allocation().unwrap();
        assert!(native.bytes() > view.try_metadata_snapshot().unwrap().nbytes());
        let expected = allocation.bytes() as u64
            + if native.identity() == allocation.identity() {
                0
            } else {
                native.bytes() as u64
            };
        if cfg!(feature = "metal") && device == DeviceType::Gpu {
            assert_eq!(native.identity(), allocation.identity());
        }
        let weak = Arc::downgrade(&host);
        let hook = Hook::install();
        let mut host_first = RetainedStorage::default();
        host_first.include_host(Arc::clone(&host)).unwrap();
        host_first.include_host(Arc::clone(&host)).unwrap();
        host_first.include_array(&view).unwrap();
        assert_eq!(host_first.hosts.len(), 1);
        assert_eq!(host_first.byte_bound().unwrap(), Some(expected));
        let mut array_first = RetainedStorage::default();
        array_first.include_array(&array).unwrap();
        array_first.include_host(Arc::clone(&host)).unwrap();
        assert_eq!(array_first.byte_bound().unwrap(), Some(expected));
        host_first.merge(array_first).unwrap();
        assert!(weak.upgrade().is_some());
        assert_eq!(host_first.byte_bound().unwrap(), Some(expected));
        assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
        drop(host_first);
        assert_eq!(Arc::strong_count(&host), 1);
        assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
        drop(hook);
        // The actual source stays alive throughout inspection, as it does in
        // the loaded model. Final native host destruction remains ordinary work.
        drop(host);
        assert!(weak.upgrade().is_none());
        // Explicit value access is outside the cold interval. Retained native
        // aliases remain numerically valid after inventory and host handles die.
        assert_eq!(
            view.evaluated().unwrap().try_to_vec::<i32>().unwrap(),
            [7, 17]
        );
    }
}

#[test]
fn retained_host_inspection_does_not_reclaim_unrelated_queued_owners() {
    let host = host_buffer();
    let queued = Array::from_slice(&[41_i32, 43], &[2]);
    queued.evaluated().unwrap();
    let retired = Arc::new(AtomicUsize::new(0));
    queued
        .retain_allocation_owner(Retired(Arc::clone(&retired)))
        .unwrap();
    safemlx::try_with_submission_retirement(|| drop(queued)).unwrap();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    let hook = Hook::install();
    let mut storage = RetainedStorage::default();
    storage.include_host(Arc::clone(&host)).unwrap();
    assert!(storage.byte_bound().unwrap().unwrap() > 0);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    drop(storage);
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    drop(hook);
    safemlx::memory::clear_cache().unwrap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

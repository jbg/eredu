use super::*;
use safemlx::{
    ops::indexing::TryIndexOp, Device, DeviceType, Dtype, HostTransferBuffer, HostTransferPolicy,
    Stream,
};

#[test]
fn registered_native_inventories_deduplicate_and_keep_all_physical_owners_charged() {
    for device in [DeviceType::Cpu, DeviceType::Gpu] {
        if device == DeviceType::Gpu && !cfg!(feature = "metal") {
            continue;
        }
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let values = (0..24).map(|n| n as f32 * 0.25 - 2.).collect::<Vec<_>>();
        let mut host =
            HostTransferBuffer::new(&[3, 8], Dtype::Float32, HostTransferPolicy::Transfer).unwrap();
        host.as_bytes_mut().unwrap().copy_from_slice(
            &values
                .iter()
                .flat_map(|n| n.to_ne_bytes())
                .collect::<Vec<_>>(),
        );
        let host = Arc::new(host.freeze());
        let weak_host = Arc::downgrade(&host);
        let array = host.copy_to_array(&stream).unwrap().synchronize().unwrap();
        array.evaluated().unwrap();
        let view = array.try_index_device((2.., ..), &stream).unwrap();
        view.evaluated().unwrap();
        let mut payload = Vec::with_capacity(256);
        payload.extend([3u8, 5, 7]);
        let source = Arc::new(payload);
        let weak_source = Arc::downgrade(&source);
        let buffer: Arc<[u8]> = Arc::from([11, 13, 17]);
        let weak_buffer = Arc::downgrade(&buffer);
        let mut sources = SourceStorage::default();
        sources
            .insert(source.clone(), source.capacity() as u64)
            .unwrap();
        let inventory = |value: &Array| {
            let mut storage = RetainedStorage::default();
            storage.include_array(value).unwrap();
            storage.include_host(host.clone()).unwrap();
            storage.include_sources(Some(sources.clone())).unwrap();
            storage.include_bytes(buffer.clone());
            storage
        };
        let first = inventory(&array);
        let second = inventory(&view);
        let expected = first.byte_bound().unwrap().unwrap();
        let host_info = host.allocation_info().unwrap();
        let native_info = array.allocation_info().unwrap().unwrap();
        assert_eq!(
            expected,
            source.capacity() as u64
                + buffer.len() as u64
                + host_info.bytes() as u64
                + if native_info.identity() == host_info.identity() {
                    0
                } else {
                    native_info.bytes() as u64
                }
        );
        if device == DeviceType::Gpu {
            assert_eq!(native_info.identity(), host_info.identity());
        }
        let physical = first
            .storage_entries()
            .unwrap()
            .iter()
            .map(|(_, bytes)| bytes)
            .sum::<u64>();
        let rows = first.storage_entries().unwrap().len();
        let controls = 2 * crate::memory_fixture::publication_control_bytes(rows);
        let pool = crate::memory_fixture::ledger(physical + controls, 0).unwrap();
        let first = first.register(&pool).unwrap();
        let second = second.register(&pool).unwrap();
        assert_eq!(first.bytes(), Some(physical));
        assert_eq!(pool.fixture_host_charge().unwrap(), physical);
        assert!(matches!(
            pool.register_host_storage([(1u8, 1)]),
            Err(WorkingMemoryError::Domain(
                eredu_core::MemoryDomainError::BudgetExceeded {
                    limit_bytes,
                    existing_bytes,
                    ..
                }
            ))
        ));
        drop((source, sources, host, buffer, array, view));
        let clone = first.clone();
        drop((first, second));
        assert!(weak_host.upgrade().is_some());
        assert!(weak_source.upgrade().is_some());
        assert!(weak_buffer.upgrade().is_some());
        assert_eq!(
            clone
                .inventory()
                .arrays
                .values()
                .next()
                .unwrap()
                .owned()
                .unwrap()
                .1
                .evaluated()
                .unwrap()
                .as_slice::<f32>(),
            values
        );
        assert_eq!(pool.fixture_host_charge().unwrap(), physical);
        drop(clone);
        assert_eq!(pool.fixture_host_charge().unwrap(), physical);
        crate::backend::ordinary_retirement::reclaim_all();
        assert!(weak_host.upgrade().is_none());
        assert!(weak_source.upgrade().is_none());
        assert!(weak_buffer.upgrade().is_none());
        assert_eq!(pool.fixture_host_charge().unwrap(), 0);
        assert_eq!(pool.fixture_host_peak().unwrap(), physical + controls);
    }
}

#[test]
fn native_storage_registration_rejects_unknown_and_conflicting_bounds_without_work() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let array = Array::from_slice(&[3.0f32, 5.], &[2]);
    array.evaluated().unwrap();
    let mut storage = RetainedStorage::default();
    storage.include_array(&array).unwrap();
    let bytes = storage
        .storage_entries()
        .unwrap()
        .iter()
        .map(|(_, bytes)| bytes)
        .sum::<u64>();
    let controls =
        crate::memory_fixture::publication_control_bytes(storage.storage_entries().unwrap().len());
    let pool = crate::memory_fixture::ledger(bytes + 2 * controls, 0).unwrap();
    let owner = storage.register(&pool).unwrap();
    let lazy = array.square(&stream).unwrap();
    let mut unknown = RetainedStorage::default();
    unknown.include_array(&lazy).unwrap();
    let error = unknown.register(&pool).unwrap_err();
    assert!(matches!(
        error,
        Error::PrefillControl(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(lazy.allocation_info().unwrap(), None);
    let mut conflict = RetainedStorage::default();
    conflict.include_array(&array).unwrap();
    let NativeEntry::Owned((conflicting_bytes, _)) = conflict.arrays.values_mut().next().unwrap()
    else {
        panic!("unpublished source")
    };
    *conflicting_bytes += 1;
    let error = conflict.register(&pool).unwrap_err();
    assert!(matches!(
        error,
        Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
    ));
    assert_eq!(pool.fixture_host_charge().unwrap(), bytes);
    assert_eq!(pool.fixture_host_peak().unwrap(), bytes + controls);
    drop(owner);
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[test]
fn registered_storage_keeps_its_charge_while_native_lock_defers_cleanup() {
    let mut host =
        HostTransferBuffer::new(&[2], Dtype::Int32, HostTransferPolicy::Transfer).unwrap();
    host.as_bytes_mut().unwrap().copy_from_slice(
        &[3i32, 5]
            .into_iter()
            .flat_map(i32::to_ne_bytes)
            .collect::<Vec<_>>(),
    );
    let host = Arc::new(host.freeze());
    let weak = Arc::downgrade(&host);
    let mut storage = RetainedStorage::default();
    storage.include_host(host).unwrap();
    let bytes = storage
        .storage_entries()
        .unwrap()
        .iter()
        .map(|(_, bytes)| bytes)
        .sum::<u64>();
    let controls =
        crate::memory_fixture::publication_control_bytes(storage.storage_entries().unwrap().len());
    let pool = crate::memory_fixture::ledger(bytes + controls, 0).unwrap();
    let owner = storage.register(&pool).unwrap();
    let (locked_tx, locked_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || loop {
        if safemlx::try_with_submission_retirement(|| {
            locked_tx.send(()).unwrap();
            release_rx
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap();
        })
        .is_some()
        {
            break;
        }
        std::thread::yield_now();
    });
    locked_rx.recv().unwrap();
    drop(owner);
    // Dropping the registration only stages cleanup. Explicit synchronous host
    // reclamation may wait for other runtime users, so perform it after release.
    assert!(weak.upgrade().is_some());
    assert_eq!(pool.fixture_host_charge().unwrap(), bytes);
    assert!(pool.register_host_storage([(1u8, 1)]).is_err());
    release_tx.send(()).unwrap();
    worker.join().unwrap();
    crate::backend::ordinary_retirement::reclaim_all();
    assert!(weak.upgrade().is_none());
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
    assert_eq!(pool.fixture_host_peak().unwrap(), bytes + controls);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::{FundingFixture as _, StorageFixture as _};

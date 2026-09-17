use super::*;
use safemlx::{
    ops::indexing::TryIndexOp, Device, DeviceType, Dtype, HostTransferBuffer, HostTransferPolicy,
    Stream,
};

fn devices() -> Vec<DeviceType> {
    if cfg!(feature = "metal") {
        vec![DeviceType::Cpu, DeviceType::Gpu]
    } else {
        vec![DeviceType::Cpu]
    }
}

fn reclaim(pool: &WorkingMemoryPool, bytes: u64, stream: &Stream) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        stream.synchronize().unwrap();
        crate::backend::ordinary_retirement::reclaim_all();
        safemlx::reclaim_allocation_owners();
        if pool.used_bytes().unwrap() == bytes {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "published allocation charge did not retire"
        );
        std::thread::yield_now();
    }
}

#[test]
fn publication_charges_independent_backings_and_aliases_without_root_cycles() {
    for device in devices() {
        let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
        let owner = NativeMemoryOwner::acquire(&pool).unwrap();
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let first = Array::from_slice(&[3.0_f32, 5., 7., 9.], &[4]);
        let second = Array::from_slice(&vec![11.0_f32; 16384], &[16384]);
        first.evaluated().unwrap();
        second.evaluated().unwrap();
        let view = first.try_index_device(1.., &stream).unwrap();
        view.evaluated().unwrap();
        let first_bytes = first.allocation_info().unwrap().unwrap().bytes() as u64;
        let second_bytes = second.allocation_info().unwrap().unwrap().bytes() as u64;
        assert_ne!(first_bytes, second_bytes);
        let mut storage = RetainedStorage::default();
        for array in [&first, &view, &second] {
            storage.include_array(array).unwrap();
        }
        let publication = storage.publish_unquoted(&owner).unwrap();
        let mut duplicate = RetainedStorage::default();
        duplicate.include_array(&view).unwrap();
        let duplicate = duplicate.publish_unquoted(&owner).unwrap();
        let total = first_bytes + second_bytes;
        assert_eq!(pool.used_bytes().unwrap(), total);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        drop((publication, duplicate, first));
        reclaim(&pool, total, &stream);
        assert_eq!(view.evaluated().unwrap().as_slice::<f32>(), &[5., 7., 9.]);
        drop(view);
        reclaim(&pool, second_bytes, &stream);
        assert_eq!(second.evaluated().unwrap().as_slice::<f32>()[0], 11.);
        drop(second);
        reclaim(&pool, 0, &stream);
        assert_eq!(pool.peak_bytes().unwrap(), total);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        drop(owner);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }
}

#[test]
fn host_publication_survives_array_aliases_until_final_physical_owner() {
    for device in devices() {
        let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
        let owner = NativeMemoryOwner::acquire(&pool).unwrap();
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let mut host =
            HostTransferBuffer::new(&[4], Dtype::Float32, HostTransferPolicy::Transfer).unwrap();
        host.as_bytes_mut().unwrap().copy_from_slice(
            &[3_f32, 5., 7., 9.]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect::<Vec<_>>(),
        );
        let host = Arc::new(host.freeze());
        let host_bytes = host.allocation_info().unwrap().bytes() as u64;
        let mut only_host = RetainedStorage::default();
        only_host.include_host(host.clone()).unwrap();
        let only_host = only_host.publish_unquoted(&owner).unwrap();
        let array = host.copy_to_array(&stream).unwrap().synchronize().unwrap();
        array.evaluated().unwrap();
        let native = array.allocation_info().unwrap().unwrap();
        let shared = native.identity() == host.allocation_info().unwrap().identity();
        if device == DeviceType::Gpu {
            assert!(shared);
        }
        let view = array.try_index_device(1.., &stream).unwrap();
        view.evaluated().unwrap();
        let mut combined = RetainedStorage::default();
        combined.include_host(host.clone()).unwrap();
        combined.include_array(&array).unwrap();
        combined.include_array(&view).unwrap();
        let combined = combined.publish_unquoted(&owner).unwrap();
        let total = host_bytes + if shared { 0 } else { native.bytes() as u64 };
        assert_eq!(pool.used_bytes().unwrap(), total);
        drop((only_host, combined, array, view));
        reclaim(&pool, host_bytes, &stream);
        drop(host);
        reclaim(&pool, 0, &stream);
        assert_eq!(pool.peak_bytes().unwrap(), total);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
    }
}

#[test]
fn non_native_publication_retires_payloads_outside_native_lock_before_charges() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Payload(Arc<AtomicUsize>);
    impl Drop for Payload {
        fn drop(&mut self) {
            assert!(safemlx::can_reclaim_submission_resources());
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let pool = WorkingMemoryPool::new(4096, 0).unwrap();
    let owner = NativeMemoryOwner::acquire(&pool).unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let source = Arc::new(Payload(drops.clone()));
    let weak_source = Arc::downgrade(&source);
    let buffer: Arc<[u8]> = Arc::from([3_u8, 5, 7]);
    let weak_buffer = Arc::downgrade(&buffer);
    let mut sources = SourceStorage::default();
    sources.insert(source, 17).unwrap();
    let mut storage = RetainedStorage::default();
    storage.include_sources(Some(sources)).unwrap();
    storage.include_bytes(buffer);
    let publication = storage.publish_unquoted(&owner).unwrap();
    let alias = publication.clone();
    drop(publication);
    assert_eq!(pool.used_bytes().unwrap(), 20);
    assert!(weak_source.upgrade().is_some());
    assert!(weak_buffer.upgrade().is_some());
    drop(alias);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(pool.used_bytes().unwrap(), 20);
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(weak_source.upgrade().is_none());
    assert!(weak_buffer.upgrade().is_none());
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.peak_bytes().unwrap(), 20);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
}

#[test]
fn publication_rejects_complete_overcapacity_batch_and_unknown_lazy_storage_atomically() {
    let pool = WorkingMemoryPool::new(1 << 14, 0).unwrap();
    let owner = NativeMemoryOwner::acquire(&pool).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let first = Array::from_slice(&[3_f32, 5.], &[2]);
    let second = Array::from_slice(&vec![7_f32; 16384], &[16384]);
    first.evaluated().unwrap();
    second.evaluated().unwrap();
    assert!(first.allocation_info().unwrap().unwrap().bytes() <= 1 << 14);
    assert!(second.allocation_info().unwrap().unwrap().bytes() > 1 << 14);
    let mut storage = RetainedStorage::default();
    storage.include_array(&first).unwrap();
    storage.include_array(&second).unwrap();
    let Error::Other(error) = storage.publish_unquoted(&owner).unwrap_err() else {
        panic!("typed capacity error")
    };
    assert!(matches!(
        error.downcast_ref::<WorkingMemoryError>(),
        Some(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.peak_bytes().unwrap(), 0);
    let lazy = first.square(&stream).unwrap();
    let mut unknown = RetainedStorage::default();
    unknown.include_array(&second).unwrap();
    unknown.include_array(&lazy).unwrap();
    let Error::Other(error) = unknown.publish_unquoted(&owner).unwrap_err() else {
        panic!("typed unknown bound")
    };
    assert_eq!(
        error.downcast_ref::<WorkingMemoryError>(),
        Some(&WorkingMemoryError::UnknownBound)
    );
    assert_eq!(lazy.allocation_info().unwrap(), None);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(pool.peak_bytes().unwrap(), 0);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
}

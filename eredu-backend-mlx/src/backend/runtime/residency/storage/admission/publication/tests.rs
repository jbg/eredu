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

fn reclaim(pool: &MemoryLedger, bytes: u64, stream: &Stream) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        stream.synchronize().unwrap();
        crate::backend::ordinary_retirement::reclaim_all();
        safemlx::memory::clear_cache().unwrap();
        safemlx::reclaim_allocation_owners();
        if pool.fixture_host_charge().unwrap() == bytes {
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
        let pool = crate::memory_fixture::ledger(1 << 24, 0).unwrap();
        let owner = NativeMemoryOwner::acquire(&pool).unwrap();
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let first = Array::from_slice(&[3.0_f32, 5., 7., 9.], &[4]);
        let second = Array::from_slice(&vec![11.0_f32; 16384], &[16384]);
        first.evaluated().unwrap();
        second.evaluated().unwrap();
        let view = first.try_index_device(1.., &stream).unwrap();
        view.evaluated().unwrap();
        let first_bytes = {
            let facts = first.allocation_info().unwrap().unwrap();
            (facts.bytes() + facts.host_control_bytes()) as u64
        };
        let second_bytes = {
            let facts = second.allocation_info().unwrap().unwrap();
            (facts.bytes() + facts.host_control_bytes()) as u64
        };
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
        assert_eq!(pool.fixture_host_charge().unwrap(), total);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        drop((publication, duplicate, first));
        reclaim(&pool, total, &stream);
        assert_eq!(view.evaluated().unwrap().as_slice::<f32>(), &[5., 7., 9.]);
        drop(view);
        reclaim(&pool, second_bytes, &stream);
        assert_eq!(second.evaluated().unwrap().as_slice::<f32>()[0], 11.);
        drop(second);
        reclaim(&pool, 0, &stream);
        assert_eq!(
            pool.fixture_host_peak().unwrap(),
            total
                + crate::memory_fixture::publication_control_bytes(4)
                + crate::memory_fixture::publication_control_bytes(2)
                + crate::memory_fixture::native_owner_control_bytes()
        );
        assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
        drop(owner);
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    }
}

#[test]
fn host_publication_survives_array_aliases_until_final_physical_owner() {
    for device in devices() {
        let pool = crate::memory_fixture::ledger(1 << 24, 0).unwrap();
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
        let host_facts = host.allocation_info().unwrap();
        let host_bytes = (host_facts.bytes() + host_facts.host_control_bytes()) as u64;
        let mut only_host = RetainedStorage::default();
        only_host.include_host(host.clone()).unwrap();
        assert_eq!(super::native::borrowed_native_roots(&only_host).count(), 1);
        let observed = host.inspect_original_source().unwrap();
        assert!(
            !observed.is_prepared_source(),
            "ordinary constructor has positive ordinary provenance"
        );
        drop(observed);
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
        let roots = super::native::borrowed_native_roots(&combined).collect::<Vec<_>>();
        assert_eq!(roots.len(), if shared { 1 } else { 2 });
        assert_eq!(
            roots
                .iter()
                .filter(|root| matches!(
                    root,
                    super::super::super::native_storage::NativeStorageRoot::Host(_, _)
                ))
                .count(),
            1
        );
        assert_eq!(
            roots
                .iter()
                .filter(|root| matches!(
                    root,
                    super::super::super::native_storage::NativeStorageRoot::Array(_)
                ))
                .count(),
            usize::from(!shared)
        );
        drop(roots);
        let combined = combined.publish_unquoted(&owner).unwrap();
        let total = host_bytes
            + if shared {
                0
            } else {
                (native.bytes() + native.host_control_bytes()) as u64
            };
        assert_eq!(pool.fixture_host_charge().unwrap(), total);
        drop((only_host, combined, array, view));
        reclaim(&pool, host_bytes, &stream);
        drop(host);
        reclaim(&pool, 0, &stream);
        assert_eq!(
            pool.fixture_host_peak().unwrap(),
            total
                + crate::memory_fixture::publication_control_bytes(2)
                + crate::memory_fixture::publication_control_bytes(if shared { 2 } else { 4 })
                + crate::memory_fixture::native_owner_control_bytes()
        );
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
    let pool = crate::memory_fixture::ledger(1 << 20, 0).unwrap();
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
    assert_eq!(pool.fixture_host_charge().unwrap(), 20);
    assert!(weak_source.upgrade().is_some());
    assert!(weak_buffer.upgrade().is_some());
    drop(alias);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(pool.fixture_host_charge().unwrap(), 20);
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(weak_source.upgrade().is_none());
    assert!(weak_buffer.upgrade().is_none());
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
    assert_eq!(
        pool.fixture_host_peak().unwrap(),
        20 + crate::memory_fixture::publication_control_bytes(2)
            + crate::memory_fixture::native_owner_control_bytes()
    );
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
}

#[test]
fn publication_retirement_requires_exact_retained_source_charges() {
    fn publish(pool: &MemoryLedger, bytes: &[Arc<[u8]>]) -> RetainedStoragePublication {
        let owner = NativeMemoryOwner::acquire(pool).unwrap();
        let mut inventory = RetainedStorage::default();
        for source in bytes {
            inventory.include_bytes(source.clone());
        }
        inventory.publish_unquoted(&owner).unwrap()
    }
    let pool = crate::memory_fixture::ledger(1 << 20, 0).unwrap();
    let foreign_pool = crate::memory_fixture::ledger(1 << 20, 0).unwrap();
    let source: Arc<[u8]> = Arc::from([3, 5, 7]);
    let other: Arc<[u8]> = Arc::from([3, 5, 7]);
    let initial = publish(&pool, &[source.clone()]);
    let later = publish(&pool, &[source.clone()]);
    let foreign = publish(&foreign_pool, &[source.clone()]);
    let changed = publish(&pool, &[other.clone()]);
    let grown = publish(&pool, &[source.clone(), other.clone()]);
    assert!(later.can_retire_with(&initial));
    assert!(!later.can_retire_with(&foreign));
    assert!(!changed.can_retire_with(&initial));
    assert!(!grown.can_retire_with(&initial));
    assert_eq!(pool.fixture_host_charge().unwrap(), 6);
    drop(later);
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(pool.fixture_host_charge().unwrap(), 6);
    drop((changed, grown, other));
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(pool.fixture_host_charge().unwrap(), 3);
    drop((initial, source));
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
    assert_eq!(foreign_pool.fixture_host_charge().unwrap(), 3);
    drop(foreign);
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(foreign_pool.fixture_host_charge().unwrap(), 0);
}

#[test]
fn publication_rejects_complete_overcapacity_batch_and_unknown_lazy_storage_atomically() {
    let controls = crate::memory_fixture::publication_control_bytes(4);
    let pool = crate::memory_fixture::ledger(
        (1 << 14) + controls + crate::memory_fixture::native_owner_control_bytes(),
        0,
    )
    .unwrap();
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
        Some(WorkingMemoryError::Domain(
            eredu_core::MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
    assert_eq!(
        pool.fixture_host_peak().unwrap(),
        controls + crate::memory_fixture::native_owner_control_bytes()
    );
    let lazy = first.square(&stream).unwrap();
    let mut unknown = RetainedStorage::default();
    unknown.include_array(&second).unwrap();
    unknown.include_array(&lazy).unwrap();
    let error = unknown.publish_unquoted(&owner).unwrap_err();
    assert!(matches!(
        error,
        Error::PrefillControl(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(lazy.allocation_info().unwrap(), None);
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
    assert_eq!(
        pool.fixture_host_peak().unwrap(),
        controls + crate::memory_fixture::native_owner_control_bytes()
    );
    assert_eq!(pool.unquoted_owner_count().unwrap(), 1);
}

#[test]
fn completed_attachment_receipts_match_exact_backing_and_domain_without_pinning_payload() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let pool = crate::memory_fixture::ledger(1 << 24, 0).unwrap();
    let foreign = crate::memory_fixture::ledger(1 << 24, 0).unwrap();
    let owner = NativeMemoryOwner::acquire(&pool).unwrap();
    let root = Array::from_slice(&[19u32, 23], &[2]);
    let alias = root.clone();
    let facts = root.allocation_info().unwrap().unwrap();
    let mut storage = RetainedStorage::default();
    storage.include_array(&root).unwrap();
    assert!(matches!(
        storage.arrays.get(&facts.identity()),
        Some(NativeEntry::Owned(_))
    ));
    let receipt = storage.publish_unquoted(&owner).unwrap();
    assert!(receipt.has_native_attachment(pool.shared_storage_accounting_id(), facts));
    assert!(!receipt.has_native_attachment(foreign.shared_storage_accounting_id(), facts));
    assert!(!receipt.has_native_attachment_facts(
        pool.shared_storage_accounting_id(),
        facts.identity(),
        facts.bytes() as u64 + 1
    ));
    assert_eq!(receipt.0._non_native.array_entries().count(), 0);
    assert!(
        receipt.0._registrations.is_empty(),
        "native charge lives on its backing"
    );
    drop(root);
    reclaim(
        &pool,
        (facts.bytes() + facts.host_control_bytes()) as u64,
        &stream,
    );
    assert_eq!(alias.evaluated().unwrap().as_slice::<u32>(), &[19, 23]);
    drop(alias);
    reclaim(&pool, 0, &stream);
    // The receipt outlives physical retirement but its never-reused generation
    // cannot cover a new allocation, even one with exactly the same capacity.
    let next = Array::from_slice(&[29u32, 31], &[2]);
    let next_facts = next.allocation_info().unwrap().unwrap();
    assert_eq!(next_facts.bytes(), facts.bytes());
    assert_ne!(next_facts.identity(), facts.identity());
    assert!(!receipt.has_native_attachment(pool.shared_storage_accounting_id(), next_facts));
    drop((next, receipt, owner));
    reclaim(&pool, 0, &stream);
}

#[test]
fn completed_host_receipt_keeps_only_exact_domain_generation_and_capacity() {
    let pool = crate::memory_fixture::ledger(1 << 24, 0).unwrap();
    let foreign = crate::memory_fixture::ledger(1 << 24, 0).unwrap();
    let owner = NativeMemoryOwner::acquire(&pool).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let host = Arc::new(
        HostTransferBuffer::new(&[4], Dtype::Float32, HostTransferPolicy::Transfer)
            .unwrap()
            .freeze(),
    );
    let weak = Arc::downgrade(&host);
    let facts = host.allocation_info().unwrap();
    let mut storage = RetainedStorage::default();
    storage.include_host(host.clone()).unwrap();
    let receipt = storage.publish_unquoted(&owner).unwrap();
    assert!(receipt.has_native_attachment(pool.shared_storage_accounting_id(), facts));
    assert!(!receipt.has_native_attachment(foreign.shared_storage_accounting_id(), facts));
    assert!(!receipt.has_native_attachment_facts(
        pool.shared_storage_accounting_id(),
        facts.identity(),
        facts.bytes() as u64 + 1,
    ));
    assert_eq!(receipt.0._non_native.host_entries().count(), 0);
    assert_eq!(receipt.0._non_native.byte_bound().unwrap(), Some(0));
    assert!(receipt.0._registrations.is_empty());
    drop(host);
    reclaim(&pool, 0, &stream);
    assert!(
        weak.upgrade().is_none(),
        "scalar receipts never pin the Host Arc"
    );
    assert!(receipt.has_native_attachment(pool.shared_storage_accounting_id(), facts));
    drop((receipt, owner));
    reclaim(&pool, 0, &stream);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;

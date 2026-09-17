#[test]
fn retained_storage_survives_manager_eviction_and_preserves_physical_capacity() {
    let (dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        [single("a", "a")],
    );
    manager.initialize().unwrap();
    assert_eq!(
        manager.retained_storage().unwrap().byte_bound().unwrap(),
        Some(0)
    );
    manager.prefetch(&id("a"), MemoryTier::Host).unwrap();
    manager.prefetch(&id("a"), MemoryTier::Device).unwrap();
    let (host_capacity, device_capacity, weak_host) = {
        let state = manager.inner.state.lock().unwrap();
        let unit = &state.storage[&id("a")];
        let host = &unit.host.as_ref().unwrap().buffers["weight"];
        let device = &unit.device.as_ref().unwrap().arrays["weight"];
        (
            host.capacity().unwrap() as u64,
            device.allocation_info().unwrap().unwrap().bytes() as u64,
            host.observe_for_test(),
        )
    };
    let report = manager.report().unwrap();
    std::fs::remove_file(dir.path().join("model.safetensors")).unwrap();
    let mut inventory = manager.retained_storage().unwrap();
    inventory
        .merge(manager.clone().retained_storage().unwrap())
        .unwrap();
    assert_eq!(
        inventory.byte_bound().unwrap(),
        Some(host_capacity + device_capacity)
    );
    assert_eq!(manager.report().unwrap(), report);
    assert!(manager.evict(&id("a"), MemoryTier::Host).unwrap());
    assert!(manager.evict(&id("a"), MemoryTier::Device).unwrap());
    assert_eq!(
        manager.retained_storage().unwrap().byte_bound().unwrap(),
        Some(0)
    );
    drop(manager);
    assert_eq!(
        inventory.byte_bound().unwrap(),
        Some(host_capacity + device_capacity)
    );
    assert_eq!(
        weak_host.upgrade().unwrap().as_bytes().unwrap(),
        [1i32, 2]
            .into_iter()
            .flat_map(i32::to_ne_bytes)
            .collect::<Vec<_>>()
    );
    drop(inventory);
    assert!(weak_host.upgrade().is_none());
}

#[test]
fn retained_storage_does_not_settle_a_submitted_transfer() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        [single("a", "a")],
    );
    manager.initialize().unwrap();
    let mut transfer = manager
        .acquire_many_with_transfer(&[(id("a"), 1)], MemoryTier::Device)
        .unwrap();
    let inventory = manager.retained_storage().unwrap();
    assert_eq!(inventory.byte_bound().unwrap(), None);
    assert!(manager
        .inner
        .state
        .lock()
        .unwrap()
        .control
        .ledger()
        .copy_status(&id("a"), MemoryTier::Device)
        .unwrap()
        .unwrap()
        .in_flight()
        .is_some());
    transfer.synchronize().unwrap();
    assert!(manager
        .retained_storage()
        .unwrap()
        .byte_bound()
        .unwrap()
        .is_some());
    assert_eq!(inventory.byte_bound().unwrap(), None);
    drop(inventory);
    drop(transfer);
    manager
        .inner
        .failed_transfer
        .store(true, std::sync::atomic::Ordering::Release);
    assert_eq!(
        manager.retained_storage().unwrap().byte_bound().unwrap(),
        None
    );
    manager
        .inner
        .failed_transfer
        .store(false, std::sync::atomic::Ordering::Release);
}

#[test]
fn retained_storage_includes_exact_unit_sources_once() {
    use eredu_checkpoint::store::MemoryWeightStore;
    let memory = |value: i32, capacity| {
        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend([value, -value].into_iter().flat_map(i32::to_ne_bytes));
        let capacity = bytes.capacity() as u64;
        let source: eredu_checkpoint::store::SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([("a".into(), Dtype::I32, vec![2], bytes)])
                .unwrap(),
        );
        (source, capacity)
    };
    let (primary, primary_capacity) = memory(3, 4096);
    let (extension, extension_capacity) = memory(7, 8192);
    let manager = ResidencyManager::new_shared_sources(
        Arc::clone(&primary),
        BTreeMap::from([(id("alias"), primary), (id("extension"), extension)]),
        OffloadPlan::new(
            OffloadConfig::new(Some(24), None, 1).unwrap(),
            [
                spec("target", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("alias", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
                spec("extension", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            ],
        )
        .unwrap(),
        [
            single("target", "a"),
            single("alias", "a"),
            single("extension", "a"),
        ],
        cpu_stream(),
        cpu_stream(),
    )
    .unwrap();
    assert_eq!(
        manager.retained_storage().unwrap().byte_bound().unwrap(),
        Some(primary_capacity + extension_capacity)
    );
    manager.initialize().unwrap();
    assert_eq!(
        manager.retained_storage().unwrap().byte_bound().unwrap(),
        Some(primary_capacity + extension_capacity)
    );
}

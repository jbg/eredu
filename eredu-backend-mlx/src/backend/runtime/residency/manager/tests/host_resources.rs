#[test]
fn host_resources_follow_allocation_identity_and_preserve_the_ledger() {
    use eredu_core::resources::{ResourceCoverage, ResourceSize};
    let (_dir, store) = fixture_store();
    let make = || {
        manager(
            Arc::clone(&store),
            OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        )
    };
    let manager = make();
    manager.initialize().unwrap();
    assert!(manager
        .host_storage_resources()
        .unwrap()
        .allocations
        .is_empty());
    let lease = manager.acquire(&id("a"), MemoryTier::Host).unwrap();
    assert_eq!(host_i32(&lease, "weight"), [1, 2]);
    let before = manager.telemetry_snapshot().unwrap();
    let reads = store.source_diagnostics().unwrap().physical_reads;
    let resources = manager.host_storage_resources().unwrap();
    assert_eq!(resources, manager.host_storage_resources().unwrap());
    assert_eq!(resources.allocations.len(), 1);
    let allocation = &resources.allocations[0];
    let ResourceSize::Fixed { extent } = &allocation.size else {
        panic!()
    };
    assert_eq!(extent.payload.lower_bytes, 8);
    assert_eq!(extent.payload.upper_bytes, Some(8));
    let capacity = lease.host_value("weight").unwrap().capacity().unwrap() as u64;
    assert_eq!(extent.capacity.lower_bytes, capacity);
    assert_eq!(extent.capacity.upper_bytes, Some(capacity));
    assert!(matches!(
        resources.coverage,
        ResourceCoverage::Partial { .. }
    ));
    assert_eq!(
        manager.report().unwrap().host_storage_resources(),
        Some(&resources)
    );
    assert_eq!(manager.telemetry_snapshot().unwrap(), before);
    assert_eq!(store.source_diagnostics().unwrap().physical_reads, reads);
    assert!(matches!(
        manager.evict(&id("a"), MemoryTier::Host),
        Err(ResidencyError::Ledger(
            ResidencyLedgerError::InUseEviction { .. }
        ))
    ));

    let other = make();
    other.initialize().unwrap();
    let other_lease = other.acquire(&id("a"), MemoryTier::Host).unwrap();
    assert_ne!(
        allocation.identity,
        other.host_storage_resources().unwrap().allocations[0].identity
    );
    drop(other_lease);
    drop(lease);
    assert!(manager.evict(&id("a"), MemoryTier::Host).unwrap());
    assert!(manager
        .host_storage_resources()
        .unwrap()
        .allocations
        .is_empty());
    let replacement = manager.acquire(&id("a"), MemoryTier::Host).unwrap();
    assert_eq!(host_i32(&replacement, "weight"), [1, 2]);
    assert_ne!(
        allocation.identity,
        manager.host_storage_resources().unwrap().allocations[0].identity
    );
}

#[test]
fn host_resources_deduplicate_shared_bindings_and_keep_independent_owners() {
    let (_dir, store) = fixture_store();
    let owner = unit(
        "owner",
        [binding("weight", "a", TensorSelection::Full, 8)
            .with_logical_target("shared.weight")
            .unwrap()],
    );
    let alias = unit(
        "alias",
        [
            WeightBinding::alias("shared", "shared.weight", 8).unwrap(),
            binding("local", "b", TensorSelection::Full, 8),
        ],
    );
    let manager = manager(
        store,
        OffloadConfig::new(None, Some(16), 1).unwrap(),
        [
            spec("owner", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            spec("alias", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
        ],
        [owner, alias],
    );
    manager.initialize().unwrap();
    let lease = manager.acquire(&id("alias"), MemoryTier::Host).unwrap();
    assert_eq!(host_i32(&lease, "shared"), [1, 2]);
    assert_eq!(host_i32(&lease, "local"), [3, 4]);
    let resources = manager.host_storage_resources().unwrap();
    assert_eq!(resources.allocations.len(), 2);
    let shared = resources
        .allocations
        .iter()
        .find(|a| a.uses.len() == 2)
        .unwrap();
    assert_ne!(shared.uses[0].owner, shared.uses[1].owner);
    drop(lease);
    assert!(manager.evict(&id("alias"), MemoryTier::Host).unwrap());
    let remaining = manager.host_storage_resources().unwrap();
    assert_eq!(remaining.allocations.len(), 1);
    assert_eq!(remaining.allocations[0].identity, shared.identity);
    assert_eq!(remaining.allocations[0].uses.len(), 1);
}

#[test]
fn host_resources_transfer_holds_survive_eviction_and_failed_polling() {
    use eredu_runtime::resource_lifetimes::ResourceLifetime;
    for fail in [false, true] {
        let (_dir, store) = fixture_store();
        let manager = manager(
            store,
            OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
            [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
            [single("a", "a")],
        );
        manager.initialize().unwrap();
        manager.prefetch(&id("a"), MemoryTier::Host).unwrap();
        let cached = manager.host_storage_resources().unwrap();
        let mut transfer = manager
            .acquire_many_with_transfer(&[(id("a"), 1)], MemoryTier::Device)
            .unwrap();
        let sources = transfer.source_host_storage_resources().unwrap();
        assert_eq!(sources.resources.allocations.len(), 1);
        assert_eq!(
            sources.resources.allocations[0].identity,
            cached.allocations[0].identity
        );
        assert_eq!(
            sources.resources.allocations[0].size,
            cached.allocations[0].size
        );
        assert_ne!(
            sources.resources.allocations[0].uses,
            cached.allocations[0].uses
        );
        assert!(sources.live_at_acquire);
        assert_eq!(
            sources.lifetimes[&cached.allocations[0].identity],
            vec![ResourceLifetime::Owner(sources.resources.scope.clone())]
        );
        assert!(manager.evict(&id("a"), MemoryTier::Host).unwrap());
        assert!(manager
            .host_storage_resources()
            .unwrap()
            .allocations
            .is_empty());
        assert_eq!(
            transfer.source_host_storage_resources().unwrap().resources,
            sources.resources
        );
        let consumer = cpu_stream();
        transfer.order_after(&consumer).unwrap();
        let output = transfer.leases()[0]
            .device_value("weight")
            .unwrap()
            .add(Array::from_int(3), &consumer)
            .unwrap();
        assert_eq!(output.evaluated().unwrap().as_slice::<i32>(), [4, 5]);
        if fail {
            transfer.mark_failed_for_test();
            // A nonblocking poll can report pending under runtime contention.
            // It must never report successful completion after failure.
            assert!(!matches!(transfer.is_complete(), Ok(true)));
            assert!(transfer.synchronize().is_err());
            assert_eq!(
                transfer.source_host_storage_resources().unwrap().resources,
                sources.resources
            );
        } else {
            transfer.synchronize().unwrap();
            assert!(transfer
                .source_host_storage_resources()
                .unwrap()
                .resources
                .allocations
                .is_empty());
        }
        drop(transfer);
        MlxNeuralBackend::reclaim_retired_resources();
    }
}

#[test]
fn host_resources_for_row_ranges_track_only_the_bounded_live_cache() {
    use eredu_checkpoint::{
        recipe::DerivedWeightRecipe, rows::PreparedRowSource, store::MemoryWeightStore,
    };
    use eredu_core::residency::OffloadUnitRange;
    use eredu_runtime::RowResidencyRange;
    let source: eredu_checkpoint::store::SharedCheckpointSource = Arc::new(
        MemoryWeightStore::from_safetensors([(
            "table".into(),
            Dtype::F32,
            vec![128, 2],
            (0..256)
                .flat_map(|i| (i as f32 + 0.25).to_le_bytes())
                .collect(),
        )])
        .unwrap(),
    );
    let table = PreparedRowSource::new(
        source.clone(),
        DerivedWeightRecipe::source("table", TensorSelection::Full),
    )
    .unwrap();
    let range = OffloadUnitRange::new(id("rows"), 0, 128, 8, ResidencyPolicy::Cacheable).unwrap();
    let rows = RowResidencyRange::new(range.clone(), table, "row").unwrap();
    let plan = OffloadPlan::with_ranges(
        OffloadConfig::new(Some(16), Some(fixture_host_capacity(2)), 1).unwrap(),
        [],
        [range.clone()],
    )
    .unwrap();
    let manager = ResidencyManager::new_shared_row_ranges(
        source,
        BTreeMap::new(),
        plan,
        [],
        vec![rows],
        cpu_stream(),
        cpu_stream(),
    )
    .unwrap();
    manager.initialize().unwrap();
    let pinned = manager
        .acquire(&range.member_id(0).unwrap(), MemoryTier::Host)
        .unwrap();
    let first = manager.host_storage_resources().unwrap().allocations[0]
        .identity
        .clone();
    let mut identities = BTreeSet::new();
    for row in 1..32 {
        let lease = manager
            .acquire(&range.member_id(row).unwrap(), MemoryTier::Host)
            .unwrap();
        let bytes = lease.host_value("row").unwrap().as_bytes().unwrap();
        let values: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(values, [row as f32 * 2. + 0.25, row as f32 * 2. + 1.25]);
        let resources = manager.host_storage_resources().unwrap();
        assert_eq!(resources.allocations.len(), 2);
        assert!(resources.allocations.iter().any(|a| a.identity == first));
        let current = resources
            .allocations
            .iter()
            .find(|a| a.identity != first)
            .unwrap();
        assert!(identities.insert(current.identity.clone()));
        assert_eq!(manager.lock().unwrap().storage.len(), 2);
        drop(lease);
    }
    drop(pinned);
}

#[test]
fn detached_host_resources_survive_failed_retirement_and_observe_without_native_locks() {
    use crate::backend::submission_recovery::{wait_for_retirement, Probe, Recovery, Status};
    use eredu_runtime::detached_resources::{DetachedResourceLimits, DetachedResourceScan};
    use std::{
        cell::Cell,
        rc::Rc,
        sync::mpsc,
        time::{Duration, Instant},
    };
    struct Gate {
        done: Rc<Cell<bool>>,
        polls: Rc<Cell<usize>>,
    }
    impl Probe for Gate {
        fn seal(&mut self) {}
        fn progress(&self) -> Status {
            self.polls.set(self.polls.get() + 1);
            Status {
                settled: self.done.get(),
                failed: !self.done.get(),
                blocked: false,
            }
        }
    }
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        [single("a", "a")],
    );
    manager.initialize().unwrap();
    manager.prefetch(&id("a"), MemoryTier::Host).unwrap();
    let cached = manager.host_storage_resources().unwrap();
    let mut transfer = manager
        .acquire_many_with_transfer(&[(id("a"), 1)], MemoryTier::Device)
        .unwrap();
    let retained = transfer.retained_resources_for_test();
    let source = transfer.source_host_storage_resources().unwrap();
    assert_eq!(
        transfer.leases()[0]
            .device_value("weight")
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<i32>(),
        [1, 2]
    );
    transfer.synchronize().unwrap();
    drop(transfer);
    assert!(manager.evict(&id("a"), MemoryTier::Host).unwrap());
    // Real native source storage, with a deterministic unknown/failing retirement
    // probe: unlike a fast CPU event it cannot settle before the observation.
    let done = Rc::new(Cell::new(false));
    let polls = Rc::new(Cell::new(0));
    let recovery = Recovery::with_probe(
        retained,
        Gate {
            done: done.clone(),
            polls: polls.clone(),
        },
    );
    let failed = recovery.finish();
    assert!(failed.failed && !failed.settled);
    let limits = DetachedResourceLimits::new(16, 16).unwrap();
    let find_source = |inventory: &eredu_runtime::detached_resources::DetachedResourceInventory| {
        inventory
            .descriptions()
            .iter()
            .find(|d| d.resources.scope == source.resources.scope)
            .cloned()
    };
    let inventory = ResidencyManager::detached_storage_resources(limits).unwrap();
    assert_eq!(inventory.scan(), &DetachedResourceScan::Complete);
    assert_eq!(find_source(&inventory), Some(source.clone()));
    assert_eq!(
        source.resources.allocations[0].identity,
        cached.allocations[0].identity
    );
    assert_eq!(
        source.resources.allocations[0].size,
        cached.allocations[0].size
    );
    let polls_before_report = polls.get();
    let report = manager.report().unwrap();
    assert_eq!(
        polls.get(),
        polls_before_report,
        "residency observation advanced recovery"
    );
    assert_eq!(
        find_source(report.detached_resources().unwrap()),
        Some(source.clone())
    );
    assert!(report
        .host_storage_resources()
        .unwrap()
        .allocations
        .is_empty());

    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let holder = std::thread::spawn(move || loop {
        if safemlx::try_with_submission_retirement(|| {
            ready_tx.send(()).unwrap();
            let _ = release_rx.recv_timeout(Duration::from_secs(5));
        })
        .is_some()
        {
            break;
        }
        std::thread::yield_now();
    });
    ready_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let before = polls.get();
    let started = Instant::now();
    let observed = ResidencyManager::detached_storage_resources(limits).unwrap();
    let elapsed = started.elapsed();
    release_tx.send(()).unwrap();
    holder.join().unwrap();
    assert!(
        elapsed < Duration::from_secs(1),
        "report waited for a native lock"
    );
    assert_eq!(polls.get(), before, "report polled detached work");
    assert_eq!(find_source(&observed), Some(source.clone()));
    assert!(manager.acquire(&id("a"), MemoryTier::Device).is_err());
    done.set(true);
    let polls_before_report = polls.get();
    assert!(find_source(manager.report().unwrap().detached_resources().unwrap()).is_some());
    assert_eq!(polls.get(), polls_before_report);
    // Observation alone cannot retire even a now-terminal probe.
    assert!(find_source(&ResidencyManager::detached_storage_resources(limits).unwrap()).is_some());
    wait_for_retirement(|| {
        find_source(&ResidencyManager::detached_storage_resources(limits).unwrap()).is_none()
    });
}

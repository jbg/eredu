#[test]
fn budgets_use_deterministic_policy_then_lru_eviction() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(None, Some(16), 1).unwrap(),
        [
            spec("cache-a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            spec("window-b", 8, ResidencyPolicy::Windowed, MemoryTier::Disk),
            spec("cache-c", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
        ],
        [
            single("cache-a", "a"),
            single("window-b", "b"),
            single("cache-c", "c"),
        ],
    );
    manager.initialize().unwrap();
    manager.prefetch(&id("cache-a"), MemoryTier::Host).unwrap();
    manager.prefetch(&id("window-b"), MemoryTier::Host).unwrap();
    manager.prefetch(&id("cache-c"), MemoryTier::Host).unwrap();
    let report = manager.report().unwrap();
    assert!(state(&report, "cache-a").host_resident());
    assert!(!state(&report, "window-b").host_resident());
    assert!(state(&report, "cache-c").host_resident());
    assert_eq!(report.offload().evictions().count(), 1);
    assert_eq!(
        report.offload().evictions().bytes(),
        fixture_binding_capacity()
    );

    manager.evict(&id("cache-a"), MemoryTier::Host).unwrap();
    manager.evict(&id("cache-c"), MemoryTier::Host).unwrap();
    manager.prefetch(&id("cache-a"), MemoryTier::Host).unwrap();
    manager.prefetch(&id("cache-c"), MemoryTier::Host).unwrap();
    manager.prefetch(&id("window-b"), MemoryTier::Host).unwrap();
    let report = manager.report().unwrap();
    assert!(!state(&report, "cache-a").host_resident());
    assert!(state(&report, "cache-c").host_resident());
    assert!(state(&report, "window-b").host_resident());
    assert!(report.offload().resident_bytes().get(MemoryTier::Host) <= fixture_host_capacity(2));
}

#[test]
fn oldest_copy_is_evicted_deterministically() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(None, Some(16), 1).unwrap(),
        [
            spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            spec("b", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            spec("c", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
        ],
        [single("a", "a"), single("b", "b"), single("c", "c")],
    );
    manager.initialize().unwrap();
    manager.prefetch(&id("a"), MemoryTier::Host).unwrap();
    manager.prefetch(&id("b"), MemoryTier::Host).unwrap();
    manager.prefetch(&id("c"), MemoryTier::Host).unwrap();
    let report = manager.report().unwrap();
    assert!(!state(&report, "a").host_resident());
    assert!(state(&report, "b").host_resident());
    assert!(state(&report, "c").host_resident());
}

#[test]
fn host_and_device_budgets_are_independent() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
        [
            spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            spec("b", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
        ],
        [single("a", "a"), single("b", "b")],
    );
    manager.initialize().unwrap();
    manager.prefetch(&id("a"), MemoryTier::Host).unwrap();
    manager.prefetch(&id("a"), MemoryTier::Device).unwrap();
    manager.prefetch(&id("b"), MemoryTier::Host).unwrap();
    let report = manager.report().unwrap();
    assert!(!state(&report, "a").host_resident());
    assert!(state(&report, "a").device_resident());
    assert!(state(&report, "b").host_resident());
    assert_eq!(
        report.offload().resident_bytes().get(MemoryTier::Host),
        fixture_binding_capacity()
    );
    assert_eq!(report.offload().resident_bytes().get(MemoryTier::Device), 8);
}

#[test]
fn leases_block_eviction_and_drop_or_unwind_releases_pins() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(None, Some(8), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        [single("a", "a")],
    );
    manager.initialize().unwrap();
    let lease = manager.acquire(&id("a"), MemoryTier::Host).unwrap();
    assert!(matches!(
        manager.evict(&id("a"), MemoryTier::Host),
        Err(ResidencyError::Ledger(
            ResidencyLedgerError::InUseEviction { pin_count: 1, .. }
        ))
    ));
    drop(lease);
    assert!(manager.evict(&id("a"), MemoryTier::Host).unwrap());
    assert!(!manager.evict(&id("a"), MemoryTier::Host).unwrap());

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
        let manager = manager.clone();
        move || {
            let _lease = manager.acquire(&id("a"), MemoryTier::Host).unwrap();
            panic!("exercise lease unwinding");
        }
    }));
    assert!(result.is_err());
    assert!(manager.evict(&id("a"), MemoryTier::Host).unwrap());
}

#[test]
fn concurrent_acquisition_materializes_once_and_counts_pins() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(None, Some(8), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        [single("a", "a")],
    );
    manager.initialize().unwrap();
    let barrier = Arc::new(Barrier::new(3));
    let handles = (0..2)
        .map(|_| {
            let manager = manager.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let lease = manager.acquire(&id("a"), MemoryTier::Host).unwrap();
                barrier.wait();
                barrier.wait();
                drop(lease);
            })
        })
        .collect::<Vec<_>>();
    barrier.wait();
    let report = manager.report().unwrap();
    assert_eq!(state(&report, "a").host_pins(), 2);
    assert_eq!(
        report
            .offload()
            .transfer(TransferDirection::DiskToHost)
            .count(),
        1
    );
    assert_eq!(
        report.offload().resident_bytes().get(MemoryTier::Host),
        fixture_binding_capacity()
    );
    barrier.wait();
    for handle in handles {
        handle.join().unwrap();
    }
    assert_eq!(state(&manager.report().unwrap(), "a").host_pins(), 0);
}

#[test]
fn windows_bound_lookahead_protect_active_units_and_record_hits() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(None, Some(16), 2).unwrap(),
        [
            spec("a", 8, ResidencyPolicy::Windowed, MemoryTier::Disk),
            spec("b", 8, ResidencyPolicy::Windowed, MemoryTier::Disk),
            spec("c", 8, ResidencyPolicy::Windowed, MemoryTier::Disk),
        ],
        [single("a", "a"), single("b", "b"), single("c", "c")],
    );
    manager.initialize().unwrap();
    let first = manager
        .prepare_window(&[id("a")], &[id("a"), id("b"), id("c")], MemoryTier::Host)
        .unwrap();
    assert_eq!(first.len(), 2);
    assert!(first
        .iter()
        .all(|(_, value)| *value == PrefetchOutcome::Miss));
    let report_before = manager.report().unwrap();
    assert!(state(&report_before, "a").host_resident());
    assert!(state(&report_before, "b").host_resident());
    assert!(!state(&report_before, "c").host_resident());

    let second = manager
        .prepare_window(&[id("b")], &[id("b"), id("c")], MemoryTier::Host)
        .unwrap();
    assert_eq!(second[0].1, PrefetchOutcome::Hit);
    assert_eq!(second[1].1, PrefetchOutcome::Miss);
    let report_after = manager.report().unwrap();
    assert!(!state(&report_after, "a").host_resident());
    assert!(state(&report_after, "b").host_resident());
    assert!(state(&report_after, "c").host_resident());
    assert_eq!(report_after.active_window(), &[id("b")]);
    assert_eq!(report_after.offload().prefetch().requests(), 4);
    assert_eq!(report_after.offload().prefetch().hits(), 1);
    assert_eq!(report_after.offload().prefetch().misses(), 3);
    assert_eq!(report_before.offload().prefetch().requests(), 2);
}

#[test]
fn exhaustion_reports_pinned_in_use_and_active_blockers() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(None, Some(16), 1).unwrap(),
        [
            spec("pinned", 8, ResidencyPolicy::Pinned, MemoryTier::Host),
            spec("leased", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            spec("wanted", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
        ],
        [
            single("pinned", "a"),
            single("leased", "b"),
            single("wanted", "c"),
        ],
    );
    manager.initialize().unwrap();
    let lease = manager.acquire(&id("leased"), MemoryTier::Host).unwrap();
    let error = manager
        .prefetch(&id("wanted"), MemoryTier::Host)
        .unwrap_err();
    match error {
        ResidencyError::Ledger(ResidencyLedgerError::BudgetExhausted {
            required_bytes,
            budget_bytes,
            blocking_units,
            ..
        }) => {
            assert_eq!(required_bytes, fixture_binding_capacity());
            assert_eq!(budget_bytes, fixture_host_capacity(2));
            assert_eq!(blocking_units.len(), 2);
            assert!(blocking_units.iter().any(|unit| unit.pinned));
            assert!(blocking_units.iter().any(|unit| unit.in_use == 1));
        }
        other => panic!("unexpected error: {other}"),
    }
    drop(lease);
}

#[test]
fn demand_stalls_and_rank_local_selections_are_reported() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(None, Some(16), 1).unwrap(),
        [
            spec("range", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            spec("indices", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
        ],
        [
            unit(
                "range",
                [binding(
                    "weight",
                    "matrix",
                    TensorSelection::Range {
                        axis: 0,
                        start: 1,
                        end: 2,
                    },
                    8,
                )],
            ),
            unit(
                "indices",
                [binding(
                    "weight",
                    "matrix",
                    TensorSelection::Indices {
                        axis: 0,
                        indices: vec![2],
                    },
                    8,
                )],
            ),
        ],
    );
    manager.initialize().unwrap();
    let range = manager.acquire(&id("range"), MemoryTier::Host).unwrap();
    assert_eq!(host_i32(&range, "weight"), [12, 13]);
    drop(range);
    let indices = manager.acquire(&id("indices"), MemoryTier::Host).unwrap();
    assert_eq!(host_i32(&indices, "weight"), [14, 15]);
    let report = manager.report().unwrap();
    assert_eq!(report.offload().prefetch().stalls(), 2);
    assert_eq!(
        report.offload().resident_bytes().get(MemoryTier::Host),
        fixture_host_capacity(2)
    );
    assert_eq!(
        report.offload().peak_resident_bytes().get(MemoryTier::Host),
        fixture_host_capacity(2)
    );
    assert!(report.weight_store().cache_hits > 0);
}

#[test]
fn ordered_device_window_trims_stale_units_with_unlimited_budget() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(None, Some(24), 2).unwrap(),
        [
            spec("a", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
            spec("b", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
            spec("c", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
        ],
        [single("a", "a"), single("b", "b"), single("c", "c")],
    );
    manager.initialize().unwrap();
    let window = DeviceLayerWindow::new([id("a"), id("b"), id("c")], 2).unwrap();

    window.prepare(&manager, 0).unwrap();
    let first = manager.report().unwrap();
    assert!(state(&first, "a").device_resident());
    assert!(state(&first, "b").device_resident());
    assert!(!state(&first, "c").device_resident());

    let lease = manager.acquire(&id("b"), MemoryTier::Device).unwrap();
    window.prepare(&manager, 1).unwrap();
    let second = manager.report().unwrap();
    assert!(!state(&second, "a").device_resident());
    assert!(state(&second, "b").device_resident());
    assert!(state(&second, "c").device_resident());
    assert_eq!(state(&second, "b").device_pins(), 1);
    drop(lease);

    window.clear(&manager).unwrap();
    assert!(manager
        .report()
        .unwrap()
        .units()
        .iter()
        .all(|unit| !unit.device_resident()));
}

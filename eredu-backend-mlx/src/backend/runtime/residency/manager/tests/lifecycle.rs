#[test]
fn named_execution_groups_keep_independent_windows_and_clear_in_isolation() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(None, None, 1).unwrap(),
        [
            spec("text.0", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
            spec("text.1", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
            spec("vision.0", 8, ResidencyPolicy::Windowed, MemoryTier::Host),
        ],
        [
            single("text.0", "a"),
            single("text.1", "b"),
            single("vision.0", "c"),
        ],
    );
    manager.initialize().unwrap();
    let text = ResidentLayerGroup::new("text", [id("text.0"), id("text.1")], 1).unwrap();
    let vision = ResidentLayerGroup::new("vision", [id("vision.0")], 1).unwrap();

    text.prepare(&manager, 0).unwrap();
    vision.prepare(&manager, 0).unwrap();
    let report = manager.report().unwrap();
    assert_eq!(report.active_window(), &[id("text.0"), id("vision.0")]);
    assert!(state(&report, "text.0").device_resident());
    assert!(state(&report, "vision.0").device_resident());

    text.clear(&manager).unwrap();
    let report = manager.report().unwrap();
    assert!(!state(&report, "text.0").device_resident());
    assert!(state(&report, "vision.0").device_resident());
    assert_eq!(report.active_window(), &[id("vision.0")]);
    let vision_report = vision.report(&manager).unwrap();
    assert_eq!(vision_report.device_units(), 1);
    assert_eq!(vision_report.device_bytes(), 8);
}

fn state<'a>(report: &'a ResidencyReport, name: &str) -> &'a UnitResidencyReport {
    report
        .units()
        .iter()
        .find(|unit| unit.id() == &id(name))
        .unwrap()
}

#[test]
fn failed_batch_reservation_rolls_back_and_cache_remains_usable() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
        [
            spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            spec("b", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
        ],
        [single("a", "a"), single("b", "b")],
    );
    manager.initialize().unwrap();
    assert!(matches!(
        manager.acquire_many_with_demand(&[(id("a"), 1), (id("b"), 1)], MemoryTier::Device),
        Err(ResidencyError::Ledger(
            ResidencyLedgerError::BudgetExhausted { .. }
        ))
    ));
    let report = manager.report().unwrap();
    assert_eq!(report.offload().resident_bytes().get(MemoryTier::Device), 0);
    assert!(!state(&report, "a").device_resident());
    assert!(!state(&report, "b").device_resident());

    let lease = manager.acquire(&id("a"), MemoryTier::Device).unwrap();
    assert_eq!(lease.device_value("weight").unwrap().shape(), &[2]);
}

#[test]
fn batched_units_detach_prior_shards_at_mapping_capacity() {
    let (_dir, store) = cross_shard_store();
    let manager = manager(
        Arc::clone(&store),
        OffloadConfig::new(None, None, 1).unwrap(),
        [
            spec("left", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            spec("right", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
        ],
        [single("left", "left"), single("right", "right")],
    );
    manager.initialize().unwrap();

    let leases = manager
        .acquire_many_with_demand(&[(id("left"), 1), (id("right"), 1)], MemoryTier::Host)
        .unwrap();

    assert_eq!(host_i32(&leases[0], "weight"), [1, 2]);
    assert_eq!(host_i32(&leases[1], "weight"), [3, 4]);
    let diagnostics = store.source_diagnostics().unwrap();
    assert_eq!(diagnostics.currently_cached_shards, 1);
    assert!(diagnostics.evictions >= 1);
}

#[test]
#[ignore = "requires local MLX host-transfer device support"]
fn cross_unit_alias_reacquisition_reuses_one_pinned_owner_read() {
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
        Arc::clone(&store),
        OffloadConfig::new(None, Some(16), 1).unwrap(),
        [
            spec("owner", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            spec("alias", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
        ],
        [owner, alias],
    );
    manager.initialize().unwrap();

    let first = manager.acquire(&id("alias"), MemoryTier::Host).unwrap();
    assert_eq!(host_i32(&first, "shared"), [1, 2]);
    assert_eq!(host_i32(&first, "local"), [3, 4]);
    drop(first);
    let reads = store.source_diagnostics().unwrap().physical_reads;
    assert_eq!(reads, 2, "one owner plus one local tensor must be read");
    let report = manager.report().unwrap();
    assert_eq!(
        report
            .units()
            .iter()
            .map(|unit| unit.expected_bytes())
            .sum::<u64>(),
        16
    );

    assert!(manager.evict(&id("alias"), MemoryTier::Host).unwrap());
    let second = manager.acquire(&id("alias"), MemoryTier::Host).unwrap();
    assert_eq!(host_i32(&second, "shared"), [1, 2]);
    assert_eq!(
        store.source_diagnostics().unwrap().physical_reads,
        reads + 1,
        "only the evicted alias-local tensor may be reread; the pinned shared owner must not"
    );
}

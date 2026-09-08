#[test]
fn validates_unit_identity_bindings_sizes_and_targets() {
    let (_dir, store) = fixture_store();
    let plan = OffloadPlan::new(
        OffloadConfig::default(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
    )
    .unwrap();
    assert!(matches!(
        ResidencyManager::new(
            Arc::clone(&store),
            plan.clone(),
            [],
            cpu_stream(),
            cpu_stream()
        ),
        Err(ResidencyError::Controller(
            ResidencyControllerError::MissingUnitDefinition { .. }
        ))
    ));
    assert!(matches!(
        ResidencyManager::new(
            Arc::clone(&store),
            plan.clone(),
            [single("a", "a"), single("a", "a")],
            cpu_stream(),
            cpu_stream()
        ),
        Err(ResidencyError::Controller(
            ResidencyControllerError::DuplicateUnitDefinition { .. }
        ))
    ));
    assert!(matches!(
        ResidencyManager::new(
            Arc::clone(&store),
            plan.clone(),
            [single("a", "a"), single("b", "b")],
            cpu_stream(),
            cpu_stream()
        ),
        Err(ResidencyError::Controller(
            ResidencyControllerError::UnexpectedUnitDefinition { .. }
        ))
    ));
    assert!(matches!(
        OffloadUnit::new(id("empty"), []),
        Err(eredu_runtime::ResidencyDeclarationError::EmptyUnit { .. })
    ));
    let duplicate = binding("same", "a", TensorSelection::Full, 8);
    assert!(matches!(
        OffloadUnit::new(id("duplicate"), [duplicate.clone(), duplicate]),
        Err(eredu_runtime::ResidencyDeclarationError::DuplicateBindingName { .. })
    ));
    let wrong = unit("a", [binding("weight", "a", TensorSelection::Full, 4)]);
    assert!(matches!(
        ResidencyManager::new(
            Arc::clone(&store),
            plan.clone(),
            [wrong],
            cpu_stream(),
            cpu_stream()
        ),
        Err(ResidencyError::Controller(
            ResidencyControllerError::BindingByteMismatch { .. }
        ))
    ));

    let valid =
        ResidencyManager::new(store, plan, [single("a", "a")], cpu_stream(), cpu_stream()).unwrap();
    assert!(matches!(
        valid.prefetch(&id("a"), MemoryTier::Disk),
        Err(ResidencyError::Ledger(
            ResidencyLedgerError::InvalidTargetTier { .. }
        ))
    ));
}

#[test]
fn detects_unit_total_overflow_before_checkpoint_access() {
    let (_dir, store) = fixture_store();
    let overflowing = unit(
        "a",
        [
            binding("a", "a", TensorSelection::Full, 8),
            binding("z", "missing", TensorSelection::Full, u64::MAX),
        ],
    );
    let plan = OffloadPlan::new(
        OffloadConfig::default(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
    )
    .unwrap();
    assert!(matches!(
        ResidencyManager::new(store, plan, [overflowing], cpu_stream(), cpu_stream()),
        Err(ResidencyError::Controller(
            ResidencyControllerError::ArithmeticOverflow { .. }
        ))
    ));
}

#[test]
fn initialization_honors_planned_tiers_and_pinning() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
        [
            spec("disk", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            spec("host", 8, ResidencyPolicy::Pinned, MemoryTier::Host),
            spec("device", 8, ResidencyPolicy::Cacheable, MemoryTier::Device),
        ],
        [
            single("disk", "a"),
            single("host", "b"),
            single("device", "c"),
        ],
    );
    assert!(matches!(
        manager.acquire(&id("disk"), MemoryTier::Host),
        Err(ResidencyError::Ledger(ResidencyLedgerError::NotInitialized))
    ));
    manager.initialize().unwrap();
    let report = manager.report().unwrap();
    assert!(report.initialized());
    assert!(!state(&report, "disk").host_resident());
    assert!(state(&report, "host").host_resident());
    assert!(state(&report, "device").device_resident());
    assert_eq!(
        report.offload().resident_bytes().get(MemoryTier::Host),
        fixture_binding_capacity()
    );
    assert_eq!(report.offload().resident_bytes().get(MemoryTier::Device), 8);
    assert!(matches!(
        manager.evict(&id("host"), MemoryTier::Host),
        Err(ResidencyError::Ledger(
            ResidencyLedgerError::PinnedEviction { .. }
        ))
    ));
}

#[test]
fn host_residency_charges_physical_allocation_capacity() {
    let dir = tempfile::tempdir().unwrap();
    let values = (0..5000u32)
        .flat_map(|value| (value as f32).to_le_bytes())
        .collect::<Vec<_>>();
    serialize_to_file(
        [(
            "weight",
            TensorView::new(Dtype::F32, vec![5000], &values).unwrap(),
        )],
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
    let logical = u64::try_from(values.len()).unwrap();
    let capacity =
        safemlx::host_transfer_capacity_upper_bound(values.len(), HostTransferPolicy::Transfer)
            .map(|capacity| u64::try_from(capacity).unwrap())
            .unwrap();
    let manager = manager(
        store,
        OffloadConfig::new(None, Some(capacity), 1).unwrap(),
        [spec(
            "host",
            logical,
            ResidencyPolicy::Pinned,
            MemoryTier::Host,
        )],
        [unit(
            "host",
            [binding("weight", "weight", TensorSelection::Full, logical)],
        )],
    );
    manager.initialize().unwrap();
    let report = manager.report().unwrap();
    let unit = state(&report, "host");
    assert_eq!(unit.expected_bytes(), logical);
    assert_eq!(unit.host_allocated_bytes(), capacity);
    assert_eq!(
        report.offload().resident_bytes().get(MemoryTier::Host),
        capacity
    );
}

#[test]
fn unsupported_owner_binding_fails_preflight_before_any_payload_read() {
    let dir = tempfile::tempdir().unwrap();
    let good = [7u8, 8];
    let bad = [9u8, 10];
    serialize_to_file(
        [
            ("good", TensorView::new(Dtype::U8, vec![2], &good).unwrap()),
            (
                "unsupported",
                TensorView::new(Dtype::F8_E5M2, vec![2], &bad).unwrap(),
            ),
        ],
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
    let plan = OffloadPlan::new(
        OffloadConfig::new(None, Some(6), 1).unwrap(),
        [
            spec("a-good", 2, ResidencyPolicy::Cacheable, MemoryTier::Host),
            spec("z-bad", 4, ResidencyPolicy::Cacheable, MemoryTier::Host),
        ],
    )
    .unwrap();
    let control = ResidencyController::new(
        store.as_ref(),
        plan,
        [
            unit(
                "a-good",
                [binding("weight", "good", TensorSelection::Full, 2)],
            ),
            unit(
                "z-bad",
                [
                    binding("a-good-copy", "good", TensorSelection::Full, 2),
                    binding("z-unsupported", "unsupported", TensorSelection::Full, 2),
                ],
            ),
        ],
    )
    .unwrap();
    let result = preflight_residency_owner_bindings(store.as_ref(), &control);
    assert!(matches!(result, Err(ResidencyError::BindingPreflight(_))));
    assert_eq!(
        store.source_diagnostics().unwrap().physical_reads,
        0,
        "backend capability rejection must precede every payload lease"
    );
}

#[test]
fn materializes_promotes_and_publishes_multi_tensor_units_atomically() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(Some(16), Some(16), 1).unwrap(),
        [spec(
            "quantized",
            16,
            ResidencyPolicy::Cacheable,
            MemoryTier::Disk,
        )],
        [unit(
            "quantized",
            [
                binding("scales", "b", TensorSelection::Full, 8),
                binding("weight", "a", TensorSelection::Full, 8),
            ],
        )],
    );
    manager.initialize().unwrap();
    assert_eq!(
        manager
            .prefetch(&id("quantized"), MemoryTier::Host)
            .unwrap(),
        PrefetchOutcome::Miss
    );
    let host = manager.acquire(&id("quantized"), MemoryTier::Host).unwrap();
    assert_eq!(
        host.binding_names().collect::<Vec<_>>(),
        ["scales", "weight"]
    );
    assert_eq!(host_i32(&host, "weight"), [1, 2]);
    assert!(matches!(
        host.device_value("weight"),
        Err(ResidencyError::HostBindingIsNotArray { .. })
    ));
    assert!(matches!(
        host.host_value("unknown"),
        Err(ResidencyError::UnknownBinding { .. })
    ));
    assert!(matches!(
        host.host_value("weight").unwrap().storage_kind().unwrap(),
        HostTransferStorageKind::Cpu
            | HostTransferStorageKind::MetalShared
            | HostTransferStorageKind::CudaPinned
    ));
    drop(host);
    assert_eq!(
        manager
            .prefetch(&id("quantized"), MemoryTier::Device)
            .unwrap(),
        PrefetchOutcome::Miss
    );
    let device = manager
        .acquire(&id("quantized"), MemoryTier::Device)
        .unwrap();
    assert_eq!(device.device_value("scales").unwrap().shape(), &[2]);
    assert!(matches!(
        device.host_value("scales"),
        Err(ResidencyError::DeviceBindingIsNotHostBuffer { .. })
    ));
    let report = manager.report().unwrap();
    assert!(state(&report, "quantized").host_resident());
    assert!(state(&report, "quantized").device_resident());
    assert_eq!(
        report
            .offload()
            .transfer(TransferDirection::DiskToHost)
            .bytes(),
        16
    );
    assert_eq!(
        report
            .offload()
            .transfer(TransferDirection::HostToDevice)
            .bytes(),
        16
    );
}

#[test]
fn direct_disk_to_device_does_not_create_a_host_copy() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(Some(8), Some(0), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        [single("a", "a")],
    );
    manager.initialize().unwrap();
    manager.prefetch(&id("a"), MemoryTier::Device).unwrap();
    let report = manager.report().unwrap();
    assert!(!state(&report, "a").host_resident());
    assert!(state(&report, "a").device_resident());
    assert_eq!(
        report
            .offload()
            .transfer(TransferDirection::DiskToDevice)
            .bytes(),
        8
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn resident_units_batch_direct_reads_across_reordered_parameters() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        Arc::clone(&store),
        OffloadConfig::new(Some(24), Some(0), 1).unwrap(),
        [spec("resident", 24, ResidencyPolicy::Pinned, MemoryTier::Device)],
        [unit("resident", [
            binding("first", "c", TensorSelection::Full, 8),
            binding("second", "a", TensorSelection::Full, 8),
            binding("third", "b", TensorSelection::Full, 8),
        ])],
    );
    manager.initialize().unwrap();
    let lease = manager.acquire(&id("resident"), MemoryTier::Device).unwrap();
    for (name, expected) in [("first", [5, 6]), ("second", [1, 2]), ("third", [3, 4])] {
        assert_eq!(lease.device_value(name).unwrap().evaluated().unwrap().as_slice::<i32>(), expected);
    }
    let diagnostics = store.source_diagnostics().unwrap();
    #[cfg(unix)]
    assert_eq!(diagnostics.physical_reads, 1);
    assert_eq!(diagnostics.physical_read_bytes, 24, "the unselected matrix is not read");
    assert_eq!(diagnostics.currently_cached_shards, 0);
}

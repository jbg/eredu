use super::*;
use eredu_nn::workspace::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Debug)]
struct State {
    used: AtomicUsize,
    limit: AtomicUsize,
    retired: AtomicBool,
}
#[derive(Debug)]
struct Account(Arc<State>);
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let limit = self.0.limit.load(Ordering::SeqCst);
        self.0
            .used
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |used| {
                used.checked_add(bytes).filter(|total| *total <= limit)
            })
            .map(|_| ())
            .map_err(|used| HostMetadataFundingError::Capacity {
                required: bytes as u64,
                available: limit.checked_sub(used).unwrap_or(0) as u64,
            })
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.0.retired.store(true, Ordering::SeqCst);
    }
}
fn funding() -> (Arc<State>, HostMetadataFunding) {
    let state = Arc::new(State {
        used: AtomicUsize::new(0),
        limit: AtomicUsize::new(16 << 20),
        retired: AtomicBool::new(false),
    });
    let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
    (state, funding)
}

#[test]
fn ordinary_host_acquisition_refuses_before_pins_and_retains_warm_lease_payer() {
    let (_directory, source) = fixture_store();
    let manager = manager(
        source,
        OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Host)],
        [single("a", "a")],
    );
    manager.initialize().unwrap();
    let request = [(id("a"), 3)];
    let (state, funding) = funding();
    state
        .limit
        .store(state.used.load(Ordering::SeqCst), Ordering::SeqCst);
    assert!(matches!(
        manager.acquire_many_with_host_transfer(&request, MemoryTier::Host, &funding),
        Err(ResidencyError::HostMetadataFunding(
            HostMetadataFundingError::Capacity { .. }
        ))
    ));
    assert_eq!(
        manager
            .inner
            .state
            .lock()
            .unwrap()
            .control
            .ledger()
            .copy_status(&id("a"), MemoryTier::Host)
            .unwrap()
            .unwrap()
            .pins(),
        0
    );
    state.limit.store(16 << 20, Ordering::SeqCst);
    let mut transfer = manager
        .acquire_many_with_host_transfer(&request, MemoryTier::Host, &funding)
        .unwrap();
    transfer.synchronize().unwrap();
    assert_eq!(host_i32(&transfer.leases()[0], "weight"), [1, 2]);
    assert_eq!(
        manager
            .inner
            .state
            .lock()
            .unwrap()
            .control
            .ledger()
            .copy_status(&id("a"), MemoryTier::Host)
            .unwrap()
            .unwrap()
            .pins(),
        1
    );
    drop(funding);
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(transfer);
    crate::backend::ordinary_retirement::reclaim_all();
    assert!(state.retired.load(Ordering::SeqCst));
    assert_eq!(
        manager
            .inner
            .state
            .lock()
            .unwrap()
            .control
            .ledger()
            .copy_status(&id("a"), MemoryTier::Host)
            .unwrap()
            .unwrap()
            .pins(),
        0
    );
}

#[test]
fn ordinary_device_acquisition_reuses_named_worker_and_retains_cached_metadata() {
    let (_directory, source) = fixture_store();
    let manager = manager(
        source,
        OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Host)],
        [single("a", "a")],
    );
    manager.initialize().unwrap();
    let (state, funding) = funding();
    let mut transfer = manager
        .acquire_many_with_host_transfer(&[(id("a"), 2)], MemoryTier::Device, &funding)
        .unwrap();
    transfer.synchronize().unwrap();
    assert_eq!(
        transfer.leases()[0]
            .device_value("weight")
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<i32>(),
        &[1, 2]
    );
    assert!(manager.inner.state.lock().unwrap().storage[&id("a")]
        .device
        .as_ref()
        .unwrap()
        .arrays
        .is_prepared());
    assert!(safemlx::OriginalScopeObserver::require_current().is_err());
    drop((funding, transfer));
    crate::backend::ordinary_retirement::reclaim_all();
    assert!(
        !state.retired.load(Ordering::SeqCst),
        "cached final names still retain their actual payer"
    );
    drop(manager);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !state.retired.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
        crate::backend::submission_recovery::reap();
        safemlx::try_retire_completed_submissions().unwrap();
        safemlx::reclaim_allocation_owners();
        crate::backend::ordinary_retirement::reclaim_all();
        std::thread::yield_now();
    }
    assert!(state.retired.load(Ordering::SeqCst));
}

#[test]
#[cfg(target_vendor = "apple")]
fn ordinary_source_pins_refuse_before_mutation_and_retain_funding_through_deferred_unpin() {
    let (_directory, source) = fixture_store();
    let manager = manager(
        source,
        OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Host)],
        [single("a", "a")],
    );
    manager.initialize().unwrap();
    let source = manager
        .host_copy_workspace(
            &[id("a")],
            crate::backend::nn::workspace::NativeAllocationFacts::current_host().unwrap(),
        )
        .unwrap();
    let pins = || {
        manager
            .inner
            .state
            .lock()
            .unwrap()
            .control
            .ledger()
            .copy_status(&id("a"), MemoryTier::Host)
            .unwrap()
            .unwrap()
            .pins()
    };
    let (state, funding) = funding();
    let baseline = state.used.load(Ordering::SeqCst);
    state.limit.store(baseline, Ordering::SeqCst);
    assert!(source
        .pin_retained_sources_with_host(&manager, &funding)
        .is_err());
    assert_eq!(pins(), 0);
    let quoted = usize::try_from(source.retained_pin_control_bytes().unwrap()).unwrap();
    state
        .limit
        .store(baseline.checked_add(quoted).unwrap(), Ordering::SeqCst);
    let retained = source
        .pin_retained_sources_with_host(&manager, &funding)
        .unwrap();
    assert_eq!(state.used.load(Ordering::SeqCst), baseline + quoted);
    assert_eq!(pins(), 1);
    assert!(manager.evict(&id("a"), MemoryTier::Host).is_err());
    drop(funding);
    {
        let locked = manager.inner.state.lock().unwrap();
        drop(retained);
        assert!(!state.retired.load(Ordering::SeqCst));
        assert_eq!(
            locked
                .control
                .ledger()
                .copy_status(&id("a"), MemoryTier::Host)
                .unwrap()
                .unwrap()
                .pins(),
            1
        );
    }
    crate::backend::ordinary_retirement::reclaim_all();
    assert!(state.retired.load(Ordering::SeqCst));
    assert_eq!(pins(), 0);
    assert!(manager.evict(&id("a"), MemoryTier::Host).unwrap());
}

#[test]
#[cfg(target_vendor = "apple")]
fn ordinary_acquisition_reuses_admitted_host_backup_and_retains_destination_payers() {
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (_directory, source) = fixture_store();
    let graph = eredu_runtime::ExecutionGraph::new(
        vec![eredu_runtime::ExecutionGroupSpec::root("only")],
        "only",
    )
    .unwrap();
    let layout = eredu_runtime::execution::ExecutionUnitLayout::new(&graph, [1]).unwrap();
    let source_stream = cpu_stream();
    let device_stream = cpu_stream();
    let _runtime =
        safemlx::PrefillRootsRuntime::prepare_for_stream(&device_stream, &source_stream).unwrap();
    let capacity = fixture_host_capacity(1);
    let plan = OffloadPlan::new(
        OffloadConfig::new(Some(capacity), Some(capacity), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Device)],
    )
    .unwrap();
    let manager = ResidencyManager::prepare_original_host(
        source.into(),
        BTreeMap::new(),
        &plan,
        &[single("a", "a")],
        &["only".into()],
        &[id("a")],
        &layout,
        1,
        &BTreeSet::new(),
        None,
        &source_stream,
        &device_stream,
        &pool,
    )
    .unwrap()
    .expect("genuine admitted immutable Host backup");
    let backup = manager
        .inner
        .sources
        .prepared_host(&id("a"))
        .unwrap()
        .clone();
    assert!(manager.inner.state.lock().unwrap().storage[&id("a")]
        .host
        .is_none());
    let snapshot = manager
        .prepared_host_copy_workspace(
            &[id("a")],
            crate::backend::nn::workspace::NativeAllocationFacts::current_host().unwrap(),
        )
        .unwrap();
    let (state, funding) = funding();
    let pins = snapshot
        .pin_retained_sources_with_host(&manager, &funding)
        .unwrap();
    assert!(manager.evict(&id("a"), MemoryTier::Device).is_err());
    let request = [(id("a"), 1)];
    let mut host = manager
        .acquire_many_with_host_transfer(&request, MemoryTier::Host, &funding)
        .unwrap();
    host.synchronize().unwrap();
    assert_eq!(host_i32(&host.leases()[0], "weight"), [1, 2]);
    assert!(
        manager.inner.state.lock().unwrap().storage[&id("a")]
            .host
            .as_ref()
            .unwrap()
            .buffers["weight"]
            .ptr_eq(&backup.buffers["weight"]),
        "Host publication retains the same immutable allocation owner"
    );
    drop((host, pins));
    crate::backend::ordinary_retirement::reclaim_all();

    // A Host pin now protects the same backing while its Device destination
    // is independently replaced by the ordinary transfer worker.
    let pins = snapshot
        .pin_retained_sources_with_host(&manager, &funding)
        .unwrap();
    assert!(manager.evict(&id("a"), MemoryTier::Device).unwrap());
    let mut device = manager
        .acquire_many_with_host_transfer(&request, MemoryTier::Device, &funding)
        .unwrap();
    let before_wait = state.used.load(Ordering::SeqCst);
    state.limit.store(before_wait, Ordering::SeqCst);
    assert!(matches!(
        device.order_after(&device_stream),
        Err(ResidencyError::HostMetadataFunding(
            HostMetadataFundingError::Capacity { .. }
        ))
    ));
    assert_eq!(state.used.load(Ordering::SeqCst), before_wait);
    state.limit.store(16 << 20, Ordering::SeqCst);
    device.order_after(&device_stream).unwrap();
    device.synchronize().unwrap();
    assert_eq!(
        device.leases()[0]
            .device_value("weight")
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<i32>(),
        &[1, 2]
    );
    assert!(manager.inner.state.lock().unwrap().storage[&id("a")]
        .device
        .as_ref()
        .unwrap()
        .arrays
        .host_sources()
        .any(|source| source.ptr_eq(&backup.buffers["weight"])));
    drop((device, pins));
    crate::backend::ordinary_retirement::reclaim_all();
    assert!(manager.evict(&id("a"), MemoryTier::Device).unwrap());
    assert!(manager.evict(&id("a"), MemoryTier::Host).unwrap());

    // The manager's immutable backup also remains the actual source when no
    // cache tier is published; this lower-level acquisition creates Device only.
    let mut device = manager
        .acquire_many_with_host_transfer(&request, MemoryTier::Device, &funding)
        .unwrap();
    device.synchronize().unwrap();
    assert_eq!(
        device.leases()[0]
            .device_value("weight")
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<i32>(),
        &[1, 2]
    );
    {
        let locked = manager.inner.state.lock().unwrap();
        assert!(locked.storage[&id("a")].host.is_none());
        assert!(locked.storage[&id("a")]
            .device
            .as_ref()
            .unwrap()
            .arrays
            .host_sources()
            .any(|source| source.ptr_eq(&backup.buffers["weight"])));
    }
    assert!(safemlx::OriginalScopeObserver::require_current().is_err());
    drop((device, funding));
    crate::backend::ordinary_retirement::reclaim_all();
    assert!(!state.retired.load(Ordering::SeqCst));
    drop((snapshot, backup, manager));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !state.retired.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
        crate::backend::submission_recovery::reap();
        safemlx::try_retire_completed_submissions().unwrap();
        safemlx::reclaim_allocation_owners();
        crate::backend::ordinary_retirement::reclaim_all();
        std::thread::yield_now();
    }
    assert!(state.retired.load(Ordering::SeqCst));
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
fn transformed_host_constructor_retains_real_leaves_and_reuses_completed_backup() {
    use eredu_checkpoint::recipe::DerivedWeightRecipe;
    if !crate::tests::support::native_process::enter("transformed-host-constructor") {
        return;
    }
    crate::backend::managed_memory::configure_memory_limits(&crate::memory_fixture::limits(
        1 << 30,
    ))
    .unwrap();
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let old_cache = safemlx::memory::set_cache_limit(1 << 24).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let rates = [-1.0f32, -4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let offsets = [4.0f32, 9.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    serialize_to_file(
        [
            (
                "rates",
                TensorView::new(Dtype::F32, vec![2], &rates).unwrap(),
            ),
            (
                "offsets",
                TensorView::new(Dtype::F32, vec![2], &offsets).unwrap(),
            ),
        ],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let source = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
    let recipe = DerivedWeightRecipe::Stack {
        axis: 0,
        inputs: vec![
            DerivedWeightRecipe::NegLog {
                input: Box::new(DerivedWeightRecipe::source("rates", TensorSelection::Full)),
            },
            DerivedWeightRecipe::SubtractOne {
                input: Box::new(DerivedWeightRecipe::source(
                    "offsets",
                    TensorSelection::Full,
                )),
            },
        ],
    };
    let graph = eredu_runtime::ExecutionGraph::new(
        vec![eredu_runtime::ExecutionGroupSpec::root("only")],
        "only",
    )
    .unwrap();
    let layout = eredu_runtime::ExecutionUnitLayout::new(&graph, [1]).unwrap();
    let source_stream = cpu_stream();
    let device_stream = cpu_stream();
    let _runtime =
        safemlx::PrefillRootsRuntime::prepare_for_stream(&device_stream, &source_stream).unwrap();
    let plan = OffloadPlan::new(
        OffloadConfig::new(Some(24), Some(fixture_host_capacity(2)), 1).unwrap(),
        [spec(
            "mixed",
            24,
            ResidencyPolicy::Cacheable,
            MemoryTier::Host,
        )],
    )
    .unwrap();
    let reclaim = || {
        source_stream.synchronize().unwrap();
        device_stream.synchronize().unwrap();
        crate::backend::submission_recovery::reap();
        safemlx::try_retire_completed_submissions().unwrap();
        safemlx::memory::clear_cache().unwrap();
        safemlx::reclaim_allocation_owners();
        crate::backend::ordinary_retirement::reclaim_all();
    };
    let charge = || {
        pool.snapshot()
            .unwrap()
            .domains
            .iter()
            .map(|domain| domain.current_charge_bytes)
            .sum::<u64>()
    };
    reclaim();
    let baseline = charge();
    let accounts = pool.snapshot().unwrap().funding_accounts;
    let manager = ResidencyManager::prepare_original_host(
        source.into(),
        BTreeMap::new(),
        &plan,
        &[unit(
            "mixed",
            [
                binding("a_direct", "rates", TensorSelection::Full, 8),
                WeightBinding::from_recipe("z_transformed", recipe, 16).unwrap(),
            ],
        )],
        &["only".into()],
        &[id("mixed")],
        &layout,
        1,
        &BTreeSet::new(),
        None,
        &source_stream,
        &device_stream,
        &pool,
    )
    .unwrap()
    .expect("transformed recipe has genuine prepared leaf sources");
    assert!(safemlx::OriginalScopeObserver::require_current().is_err());
    // This is the same ordinary host retirement boundary used immediately
    // before NativeMemoryOwner acquisition by the model-loading provider.
    crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    let completed_charge = charge();
    assert!(completed_charge > baseline);
    assert_eq!(pool.snapshot().unwrap().reservations, 0);
    // The actual constructor has retired its temporary graph while the manager
    // keeps its fixed Host storage. Ordinary loading can now acquire its usual
    // exclusion lease without discarding or readmitting those source buffers.
    let load = pool.acquire_unquoted().unwrap();
    drop(load);
    assert_eq!(charge(), completed_charge);
    let backup = manager
        .inner
        .sources
        .prepared_host(&id("mixed"))
        .unwrap()
        .clone();
    let (state, funding) = funding();
    let mut host = manager
        .acquire_many_with_host_transfer(&[(id("mixed"), 1)], MemoryTier::Host, &funding)
        .unwrap();
    host.synchronize().unwrap();
    for (name, expected) in [
        ("a_direct", vec![-1.0, -4.0]),
        ("z_transformed", vec![0.0, 4.0f32.ln(), 3.0, 8.0]),
    ] {
        let actual = host.leases()[0]
            .host_value(name)
            .unwrap()
            .as_bytes()
            .unwrap();
        let (values, tail) = actual.as_chunks::<4>();
        assert!(tail.is_empty());
        assert_eq!(values.len(), expected.len());
        for (value, expected) in values.iter().zip(expected) {
            assert!((f32::from_ne_bytes(*value) - expected).abs() < 1e-6);
        }
    }
    drop(host);
    let mut device = manager
        .acquire_many_with_host_transfer(&[(id("mixed"), 1)], MemoryTier::Device, &funding)
        .unwrap();
    device.synchronize().unwrap();
    for (value, expected) in device.leases()[0]
        .device_value("z_transformed")
        .unwrap()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .zip([0.0, 4.0f32.ln(), 3.0, 8.0])
    {
        assert!((value - expected).abs() < 1e-6);
    }
    assert!(manager.inner.state.lock().unwrap().storage[&id("mixed")]
        .device
        .as_ref()
        .unwrap()
        .arrays
        .host_sources()
        .any(|source| source.ptr_eq(&backup.buffers["z_transformed"])));
    drop((device, funding, manager));
    reclaim();
    assert!(
        charge() > baseline,
        "the immutable backup still retains its constructor account"
    );
    drop(backup);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while (charge() != baseline || !state.retired.load(Ordering::SeqCst))
        && std::time::Instant::now() < deadline
    {
        reclaim();
        std::thread::yield_now();
    }
    assert!(state.retired.load(Ordering::SeqCst));
    assert_eq!(charge(), baseline);
    assert_eq!(pool.snapshot().unwrap().funding_accounts, accounts);
    safemlx::memory::set_cache_limit(old_cache).unwrap();
}

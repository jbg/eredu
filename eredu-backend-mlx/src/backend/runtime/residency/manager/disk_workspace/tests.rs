use super::*;
use eredu_checkpoint::store::{CheckpointSource, SafetensorsWeightStore, TensorSelection};
use eredu_core::residency::{OffloadConfig, OffloadPlan, OffloadUnitSpec, ResidencyPolicy};
use safemlx::{ops::indexing::TryIndexOp, Device, DeviceType};
use safetensors::tensor::{serialize_to_file, TensorView};
use std::sync::atomic::AtomicUsize;

fn id(name: &str) -> OffloadUnitId {
    OffloadUnitId::new(name).unwrap()
}
fn facts() -> NativeAllocationFacts {
    NativeAllocationFacts::current_host().unwrap()
}
pub(super) fn fixture(
    local_alias: bool,
    device: DeviceType,
) -> (
    tempfile::TempDir,
    Arc<SafetensorsWeightStore>,
    ResidencyManager,
) {
    // Shared native suites use the genuine paid initializer before ordinary streams.
    crate::tests::support::test_utils::initialize_original_sources();
    let dir = tempfile::tempdir().unwrap();
    let a = [1_i32, 2]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    let b = [3_i32, 4]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    serialize_to_file(
        [
            (
                "a",
                TensorView::new(safetensors::Dtype::I32, vec![2], &a).unwrap(),
            ),
            (
                "b",
                TensorView::new(safetensors::Dtype::I32, vec![2], &b).unwrap(),
            ),
        ],
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let source = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
    let mut owner = vec![WeightBinding::new("weight", "a", TensorSelection::Full, 8)
        .unwrap()
        .with_logical_target("shared.weight")
        .unwrap()];
    if local_alias {
        owner.push(WeightBinding::alias("same", "shared.weight", 8).unwrap());
    }
    let units = vec![
        OffloadUnit::new(id("owner"), owner).unwrap(),
        OffloadUnit::new(
            id("alias"),
            [
                WeightBinding::alias("shared", "shared.weight", 8).unwrap(),
                WeightBinding::new("local", "b", TensorSelection::Full, 8).unwrap(),
            ],
        )
        .unwrap(),
    ];
    let specs = units.iter().map(|unit| {
        OffloadUnitSpec::new(
            unit.id().clone(),
            8,
            ResidencyPolicy::Cacheable,
            MemoryTier::Disk,
        )
        .unwrap()
    });
    let plan = OffloadPlan::new(OffloadConfig::new(None, None, 1).unwrap(), specs).unwrap();
    let manager = ResidencyManager::new(
        source.clone(),
        plan,
        units,
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0)),
        Stream::new_with_device(&Device::new(device, 0)),
    )
    .unwrap();
    manager.initialize().unwrap();
    (dir, source, manager)
}
fn unknown(result: Result<DiskCopyWorkspace, DiskCopyWorkspaceError>) {
    assert!(
        matches!(
            result,
            Err(DiskCopyWorkspaceError::Unproved {
                source: WorkingMemoryError::UnknownBound,
                ..
            })
        ),
        "{result:?}"
    );
}
struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn cold_snapshot_reuses_prepared_headers_and_exposes_canonical_dependency_closure() {
    let (_dir, store, manager) = fixture(true, DeviceType::Gpu);
    let plans = manager.prepare_disk_reads(&[id("alias")]).unwrap();
    assert_eq!(plans.requested_units(), [id("alias")]);
    assert_eq!(plans.units().len(), 2, "outside-window owner is explicit");
    let before = store.source_diagnostics().unwrap();
    let snapshot = manager.disk_copy_workspace(&plans, facts()).unwrap();
    snapshot.validate_sources(&manager).unwrap();
    assert_eq!(
        store.source_diagnostics().unwrap(),
        before,
        "candidate inspection cannot re-admit or touch source metadata"
    );
    assert_eq!(before.physical_reads, 0);
    assert_eq!(
        snapshot.fresh_capacity_bytes(),
        2 * facts().buffer_capacity(8).unwrap()
    );
    assert_eq!(snapshot.current_device_bytes(), 0);
    assert_eq!(snapshot.host_staging_bytes(), 0);
    assert_eq!(snapshot.requested_units(), [id("alias")]);
    for unit in snapshot.units() {
        assert!(!unit.currently_on_device());
        assert_eq!(
            unit.fresh_capacity_bytes(),
            facts().buffer_capacity(8).unwrap()
        );
        assert_eq!(unit.id(), unit.definition().id());
        for binding in unit.bindings() {
            assert_eq!(binding.shape(), [2]);
            assert_eq!(binding.dtype(), Dtype::Int32);
            assert_eq!(binding.logical_bytes(), 8);
            assert_eq!(
                binding.output_capacity_bytes(),
                facts().buffer_capacity(8).unwrap()
            );
            assert!(binding.current_allocation().is_none());
            if binding.binding().is_alias() {
                assert_eq!(binding.owner().unit(), &id("owner"));
                assert_eq!(binding.owner().name(), "weight");
            }
        }
    }
    let retired = Arc::new(AtomicUsize::new(0));
    drop(crate::backend::ordinary_retirement::OrdinaryRetirement::new(Retired(retired.clone())));
    snapshot.validate_sources(&manager).unwrap();
    let _debug = format!("{snapshot:?}");
    assert_eq!(
        retired.load(Ordering::SeqCst),
        0,
        "snapshot must not use manager housekeeping lock"
    );
    crate::backend::ordinary_retirement::reclaim();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn retained_dispatch_reads_owners_once_and_shares_local_and_external_aliases() {
    let (_dir, store, manager) = fixture(true, DeviceType::Gpu);
    let plans = manager.prepare_disk_reads(&[id("alias")]).unwrap();
    let source = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let execution = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let mut custody = Vec::new();
    let owner = plans
        .unit(&id("owner"))
        .unwrap()
        .materialize(&BTreeMap::new(), &source, &execution, |value| {
            custody.push(value.clone())
        })
        .unwrap();
    let shared = BTreeMap::from([("shared".to_owned(), owner["weight"].clone())]);
    let aliases = plans
        .unit(&id("alias"))
        .unwrap()
        .materialize(&shared, &source, &execution, |value| {
            custody.push(value.clone())
        })
        .unwrap();
    safemlx::transforms::eval(owner.values().chain(aliases.values())).unwrap();
    execution.synchronize().unwrap();
    for (array, expected) in [
        (&owner["weight"], [1, 2]),
        (&owner["same"], [1, 2]),
        (&aliases["shared"], [1, 2]),
        (&aliases["local"], [3, 4]),
    ] {
        assert_eq!(
            array.evaluated().unwrap().try_as_slice::<i32>().unwrap(),
            expected
        );
    }
    let owner_backing = owner["weight"].allocation_info().unwrap().unwrap();
    assert_eq!(
        owner["same"].allocation_info().unwrap().unwrap(),
        owner_backing
    );
    assert_eq!(
        aliases["shared"].allocation_info().unwrap().unwrap(),
        owner_backing
    );
    assert_ne!(
        aliases["local"]
            .allocation_info()
            .unwrap()
            .unwrap()
            .identity(),
        owner_backing.identity()
    );
    assert_eq!(
        custody.len(),
        6,
        "two read inputs, two Copy outputs, two alias custody entries"
    );
    let before = store.source_diagnostics().unwrap();
    let mut called = false;
    assert!(matches!(
        plans.unit(&id("alias")).unwrap().materialize(
            &BTreeMap::new(),
            &source,
            &execution,
            |_| called = true
        ),
        Err(DiskCopyWorkspaceError::Unproved {
            source: WorkingMemoryError::IdentityMismatch,
            ..
        })
    ));
    assert!(!called);
    assert_eq!(store.source_diagnostics().unwrap(), before);
    let wrong = BTreeMap::from([("shared".into(), Array::from_slice(&[1_f32, 2.0], &[2]))]);
    assert!(matches!(
        plans
            .unit(&id("alias"))
            .unwrap()
            .materialize(&wrong, &source, &execution, |_| called = true),
        Err(DiskCopyWorkspaceError::Unproved {
            source: WorkingMemoryError::IdentityMismatch,
            ..
        })
    ));
    assert!(
        !called,
        "wrong-typed alias rejects before direct reads or custody mutation"
    );
    assert_eq!(store.source_diagnostics().unwrap(), before);
}

#[test]
fn warm_device_snapshot_is_cold_and_does_not_retain_evicted_numerical_storage() {
    let (_dir, store, manager) = fixture(false, DeviceType::Gpu);
    let plans = manager.prepare_disk_reads(&[id("alias")]).unwrap();
    let lease = manager.acquire(&id("alias"), MemoryTier::Device).unwrap();
    let retired = Arc::new(AtomicUsize::new(0));
    lease
        .device_value("local")
        .unwrap()
        .retain_allocation_owner(Retired(retired.clone()))
        .unwrap();
    let before = store.source_diagnostics().unwrap();
    let snapshot = manager.disk_copy_workspace(&plans, facts()).unwrap();
    snapshot.validate_sources(&manager).unwrap();
    assert_eq!(store.source_diagnostics().unwrap(), before);
    let allocations = snapshot
        .units()
        .iter()
        .flat_map(|unit| unit.bindings())
        .map(|binding| {
            assert!(
                binding.output_capacity_bytes()
                    >= binding.current_allocation().unwrap().bytes() as u64
            );
            let allocation = binding.current_allocation().unwrap();
            (allocation.identity(), allocation.bytes() as u64)
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(allocations.len(), 2);
    assert_eq!(
        snapshot.current_device_bytes(),
        allocations.values().sum::<u64>()
    );
    let queued = Arc::new(AtomicUsize::new(0));
    drop(crate::backend::ordinary_retirement::OrdinaryRetirement::new(Retired(queued.clone())));
    snapshot.validate_sources(&manager).unwrap();
    assert_eq!(
        queued.load(Ordering::SeqCst),
        0,
        "warm inspection must suppress native housekeeping too"
    );
    crate::backend::ordinary_retirement::reclaim();
    drop(lease);
    assert!(manager.evict(&id("alias"), MemoryTier::Device).unwrap());
    // Logical eviction releases the manager's owner; the physical cache keeps
    // the same backing and its custody until explicit allocator eviction.
    safemlx::memory::clear_cache().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while retired.load(Ordering::SeqCst) == 0 {
        crate::backend::submission_recovery::reap();
        crate::backend::ordinary_retirement::reclaim();
        safemlx::memory::clear_cache().unwrap();
        safemlx::reclaim_allocation_owners();
        assert!(
            std::time::Instant::now() < deadline,
            "metadata descriptor retained evicted backing"
        );
        std::thread::yield_now();
    }
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    assert!(matches!(
        snapshot.validate_sources(&manager),
        Err(DiskCopyWorkspaceError::Unproved {
            source: WorkingMemoryError::IdentityMismatch,
            ..
        })
    ));
    assert_eq!(
        snapshot.plans().units().len(),
        2,
        "metadata survives physical retirement"
    );
}

#[test]
fn unknown_host_inflight_failed_or_foreign_sources_never_fall_back_or_reap() {
    let (_dir, store, manager) = fixture(false, DeviceType::Gpu);
    let plans = manager.prepare_disk_reads(&[id("alias")]).unwrap();
    let mut transfer = manager
        .acquire_many_with_transfer(&[(id("owner"), 1)], MemoryTier::Device)
        .unwrap();
    let before = store.source_diagnostics().unwrap();
    unknown(manager.disk_copy_workspace(&plans, facts()));
    assert!(manager
        .inner
        .state
        .lock()
        .unwrap()
        .control
        .ledger()
        .copy_status(&id("owner"), MemoryTier::Device)
        .unwrap()
        .unwrap()
        .in_flight()
        .is_some());
    assert_eq!(store.source_diagnostics().unwrap(), before);
    transfer.synchronize().unwrap();
    drop(transfer);
    manager.disk_copy_workspace(&plans, facts()).unwrap();
    drop(manager.acquire(&id("alias"), MemoryTier::Host).unwrap());
    let before = store.source_diagnostics().unwrap();
    unknown(manager.disk_copy_workspace(&plans, facts()));
    assert_eq!(store.source_diagnostics().unwrap(), before);
    manager.inner.failed_transfer.store(true, Ordering::Release);
    unknown(manager.disk_copy_workspace(&plans, facts()));
    manager
        .inner
        .failed_transfer
        .store(false, Ordering::Release);
    let (_other_dir, _other_store, other) = fixture(false, DeviceType::Gpu);
    assert!(matches!(
        other.disk_copy_workspace(&plans, facts()),
        Err(DiskCopyWorkspaceError::Unproved {
            source: WorkingMemoryError::IdentityMismatch,
            ..
        })
    ));
    let (_cpu_dir, _cpu_store, cpu) = fixture(false, DeviceType::Cpu);
    let cpu_plans = cpu.prepare_disk_reads(&[id("owner")]).unwrap();
    unknown(cpu.disk_copy_workspace(&cpu_plans, facts()));
}

#[test]
fn warm_capacity_uses_actual_backing_and_lazy_or_wrong_geometry_stays_unproved() {
    let (_dir, store, manager) = fixture(false, DeviceType::Gpu);
    let plans = manager.prepare_disk_reads(&[id("owner")]).unwrap();
    drop(manager.acquire(&id("owner"), MemoryTier::Device).unwrap());
    let old = manager.disk_copy_workspace(&plans, facts()).unwrap();
    let cpu = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let parent = Array::from_slice(&[7_i32; 128], &[128]);
    let view = parent.try_index_device(..2, &cpu).unwrap();
    view.evaluated().unwrap();
    let backing = view.allocation_info().unwrap().unwrap();
    assert!(backing.bytes() > 8);
    manager
        .inner
        .state
        .lock()
        .unwrap()
        .storage
        .get_mut(&id("owner"))
        .unwrap()
        .device = Some(super::super::ResidentArrays::ordinary(BTreeMap::from([(
        "weight".into(),
        view,
    )])));
    let before = store.source_diagnostics().unwrap();
    let snapshot = manager.disk_copy_workspace(&plans, facts()).unwrap();
    assert_eq!(
        snapshot.units()[0].bindings()[0].output_capacity_bytes(),
        facts()
            .buffer_capacity(8)
            .unwrap()
            .max(backing.bytes() as u64)
    );
    assert!(old.validate_sources(&manager).is_err());
    assert_eq!(store.source_diagnostics().unwrap(), before);
    let lazy = safemlx::ops::zeros_dtype(&[2], Dtype::Int32, &cpu).unwrap();
    manager
        .inner
        .state
        .lock()
        .unwrap()
        .storage
        .get_mut(&id("owner"))
        .unwrap()
        .device = Some(super::super::ResidentArrays::ordinary(BTreeMap::from([(
        "weight".into(),
        lazy,
    )])));
    unknown(manager.disk_copy_workspace(&plans, facts()));
    let wrong = Array::from_slice(&[1_f32, 2.0], &[2]);
    manager
        .inner
        .state
        .lock()
        .unwrap()
        .storage
        .get_mut(&id("owner"))
        .unwrap()
        .device = Some(super::super::ResidentArrays::ordinary(BTreeMap::from([(
        "weight".into(),
        wrong,
    )])));
    assert!(matches!(
        manager.disk_copy_workspace(&plans, facts()),
        Err(DiskCopyWorkspaceError::Unproved {
            source: WorkingMemoryError::IdentityMismatch,
            ..
        })
    ));
}

#[test]
fn capacity_overflow_is_fatal_instead_of_an_unpriced_candidate() {
    assert!(matches!(
        add(u64::MAX, 1),
        Err(DiskCopyWorkspaceError::Unproved {
            source: WorkingMemoryError::Overflow,
            ..
        })
    ));
}

#[test]
fn retained_plans_release_source_wrapper_payloads_but_preserve_admitted_reads() {
    use eredu_checkpoint::{
        recipe::RecipeInferenceCache,
        store::{
            CheckpointLease, EncodedReadBatch, SharedCheckpointSource, SourceStorage, StoreError,
            TensorMetadata, TensorReadRequest, WeightStoreDiagnostics,
        },
    };

    struct PayloadSource {
        inner: Arc<SafetensorsWeightStore>,
        payload: Arc<Vec<i32>>,
    }
    impl CheckpointSource for PayloadSource {
        fn source_storage(&self) -> Result<Option<SourceStorage>, StoreError> {
            let mut storage = self.inner.source_storage()?.unwrap();
            storage.insert(
                self.payload.clone(),
                (self.payload.capacity() * std::mem::size_of::<i32>()) as u64,
            )?;
            Ok(Some(storage))
        }
        fn recipe_cache(&self) -> Option<&RecipeInferenceCache> {
            self.inner.recipe_cache()
        }
        fn source_keys(&self) -> Vec<String> {
            self.inner.source_keys()
        }
        fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.inner.source_metadata(key)
        }
        fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            self.inner.acquire_lease(request)
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            self.inner.source_diagnostics()
        }
        fn prepare_encoded_read(
            &self,
            keys: &[String],
        ) -> Result<Option<EncodedReadBatch>, StoreError> {
            self.inner.prepare_encoded_read(keys)
        }
    }

    let (_directory, store, unused_manager) = fixture(false, DeviceType::Gpu);
    drop(unused_manager);
    let store_lifetime = Arc::downgrade(&store);
    let payload = Arc::new((1..=37).collect::<Vec<i32>>());
    let payload_lifetime = Arc::downgrade(&payload);
    let source: SharedCheckpointSource = Arc::new(PayloadSource {
        inner: store,
        payload,
    });
    let source_lifetime = Arc::downgrade(&source);
    let unit = OffloadUnit::new(
        id("owner"),
        [WeightBinding::new("weight", "a", TensorSelection::Full, 8).unwrap()],
    )
    .unwrap();
    let plan = OffloadPlan::new(
        OffloadConfig::new(None, None, 1).unwrap(),
        [
            OffloadUnitSpec::new(id("owner"), 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)
                .unwrap(),
        ],
    )
    .unwrap();
    let manager = ResidencyManager::new_shared(
        source.clone(),
        plan,
        [unit],
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0)),
        Stream::new_with_device(&Device::new(DeviceType::Gpu, 0)),
    )
    .unwrap();
    manager.initialize().unwrap();
    let prepared = manager.prepare_disk_reads(&[id("owner")]).unwrap();
    let cloned = prepared.clone();
    let snapshot = manager.disk_copy_workspace(&prepared, facts()).unwrap();
    snapshot.validate_sources(&manager).unwrap();
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    drop(source);
    assert!(
        payload_lifetime.upgrade().is_some(),
        "live manager owns the original source"
    );
    drop(manager);
    assert!(
        source_lifetime.upgrade().is_none(),
        "proof retains only weak source identity"
    );
    assert!(
        payload_lifetime.upgrade().is_none(),
        "direct support cannot extend unrelated source payload lifetime"
    );
    assert!(
        store_lifetime.upgrade().is_none(),
        "admitted reads do not require the source object"
    );

    // Native execution is explicit and occurs after source-object retirement.
    let source_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let execution_stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let mut custody = Vec::new();
    let arrays = cloned
        .unit(&id("owner"))
        .unwrap()
        .materialize(
            &BTreeMap::new(),
            &source_stream,
            &execution_stream,
            |array| custody.push(array.clone()),
        )
        .unwrap();
    safemlx::transforms::eval(arrays.values()).unwrap();
    execution_stream.synchronize().unwrap();
    assert_eq!(
        arrays["weight"]
            .evaluated()
            .unwrap()
            .try_as_slice::<i32>()
            .unwrap(),
        [1, 2]
    );
    assert_eq!(snapshot.plans().units().len(), 1);
    assert_eq!(prepared.units().len(), 1);
    assert_eq!(custody.len(), 2);
}

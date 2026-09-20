use super::*;
use eredu_checkpoint::store::{CheckpointSource, SafetensorsWeightStore, TensorSelection};
use eredu_core::residency::{OffloadConfig, OffloadPlan, OffloadUnitSpec, ResidencyPolicy};
use safemlx::{Device, DeviceType, ops::indexing::TryIndexOp};
use std::sync::atomic::AtomicUsize;

fn id(name: &str) -> OffloadUnitId {
    OffloadUnitId::new(name).unwrap()
}
fn facts() -> NativeAllocationFacts {
    NativeAllocationFacts::current_host().unwrap()
}
fn fixture() -> (
    tempfile::TempDir,
    Arc<SafetensorsWeightStore>,
    ResidencyManager,
) {
    let (directory, source, old) = super::super::tests::fixture(false, DeviceType::Gpu);
    drop(old);
    let units = (0..3)
        .map(|index| {
            OffloadUnit::new(
                id(&format!("u{index}")),
                [WeightBinding::new(
                    "weight",
                    if index == 1 { "b" } else { "a" },
                    TensorSelection::Full,
                    8,
                )
                .unwrap()],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let plan = OffloadPlan::new(
        OffloadConfig::new(None, None, 1).unwrap(),
        units.iter().map(|unit| {
            OffloadUnitSpec::new(
                unit.id().clone(),
                8,
                ResidencyPolicy::Cacheable,
                MemoryTier::Disk,
            )
            .unwrap()
        }),
    )
    .unwrap();
    let manager = ResidencyManager::new(
        source.clone(),
        plan,
        units,
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0)),
        Stream::new_with_device(&Device::new(DeviceType::Gpu, 0)),
    )
    .unwrap();
    manager.initialize().unwrap();
    (directory, source, manager)
}
fn receipt(manager: &ResidencyManager) -> DiskRouteReceipt {
    let plans = manager
        .prepare_disk_reads(&[id("u0"), id("u1"), id("u2")])
        .unwrap();
    manager
        .disk_copy_workspace(&plans, facts())
        .unwrap()
        .receipt(vec![
            vec![id("u0"), id("u1")],
            vec![id("u1"), id("u2")],
            vec![id("u2")],
        ])
        .unwrap()
}
fn assert_fenced(error: ResidencyError) {
    let ResidencyError::AdmittedDiskRoute { source } = error else {
        panic!("unexpected error: {error}");
    };
    assert!(
        matches!(
            source.downcast_ref::<DiskCopyWorkspaceError>(),
            Some(DiskCopyWorkspaceError::Unproved {
                source: WorkingMemoryError::ExecutionFenced,
                ..
            })
        ),
        "{source}"
    );
}

#[test]
fn active_route_rejects_unpriced_windows_host_and_background_before_reads() {
    let (_directory, source, manager) = fixture();
    let receipt = receipt(&manager);
    let guard = receipt.activate().unwrap();
    let before = source.source_diagnostics().unwrap();
    assert_fenced(
        manager
            .acquire(&id("u0"), MemoryTier::Device)
            .err()
            .unwrap(),
    );
    assert!(
        manager
            .set_admitted_disk_window(&[id("u0"), id("u1"), id("u2")])
            .is_err()
    );
    manager
        .set_admitted_disk_window(&[id("u0"), id("u1")])
        .unwrap();
    assert_fenced(manager.acquire(&id("u0"), MemoryTier::Host).err().unwrap());
    assert_fenced(manager.prefetch(&id("u2"), MemoryTier::Device).unwrap_err());
    let background = manager.clone();
    let error = std::thread::spawn(move || {
        background
            .prefetch(&id("u0"), MemoryTier::Device)
            .unwrap_err()
    })
    .join()
    .unwrap();
    assert_fenced(error);
    assert_eq!(source.source_diagnostics().unwrap(), before);
    drop(guard);
    assert!(!manager.admitted_disk_route_active());
    drop(manager.acquire(&id("u0"), MemoryTier::Host).unwrap());
    assert!(
        receipt.activate().is_err(),
        "unexpected warm Host must reject cold reactivation"
    );
}

#[test]
fn own_pending_plan_provenance_allows_window_overlap_but_old_cache_must_retire() {
    let (_directory, source, manager) = fixture();
    let receipt = receipt(&manager);
    let _guard = receipt.activate().unwrap();
    manager
        .set_admitted_disk_window(&[id("u0"), id("u1")])
        .unwrap();
    let mut first = manager
        .acquire_many_with_transfer(&[(id("u0"), 1)], MemoryTier::Device)
        .unwrap();
    {
        let mut state = manager.inner.state.lock().unwrap();
        validate_disk_acquisition(
            &state,
            &manager.inner.sources,
            &[id("u0")],
            MemoryTier::Device,
        )
        .unwrap();
        let provenance = state
            .storage
            .get_mut(&id("u0"))
            .unwrap()
            .device_disk_route
            .clone();
        state.storage.get_mut(&id("u0")).unwrap().device_disk_route = Weak::new();
        assert!(
            validate_disk_acquisition(
                &state,
                &manager.inner.sources,
                &[id("u0")],
                MemoryTier::Device
            )
            .is_err()
        );
        state.storage.get_mut(&id("u0")).unwrap().device_disk_route = provenance;
    }
    let before_setter = source.source_diagnostics().unwrap();
    manager
        .set_admitted_disk_window(&[id("u0"), id("u1")])
        .unwrap();
    assert!(
        manager
            .inner
            .state
            .lock()
            .unwrap()
            .control
            .ledger()
            .copy_status(&id("u0"), MemoryTier::Device)
            .unwrap()
            .unwrap()
            .in_flight()
            .is_some(),
        "window preflight must not settle its own pending transfer"
    );
    assert_eq!(source.source_diagnostics().unwrap(), before_setter);
    // The next batch is allowed while an independently owned transfer in this
    // same priced window is pending; cold receipt validation stays stricter.
    assert!(receipt.validate().is_err());
    let mut second = manager
        .acquire_many_with_transfer(&[(id("u1"), 1)], MemoryTier::Device)
        .unwrap();
    first.synchronize().unwrap();
    second.synchronize().unwrap();
    drop(first);
    drop(second);
    crate::backend::ordinary_retirement::reclaim();
    manager
        .set_admitted_disk_window(&[id("u1"), id("u2")])
        .unwrap();
    let before = source.source_diagnostics().unwrap();
    assert_fenced(
        manager
            .acquire(&id("u2"), MemoryTier::Device)
            .err()
            .unwrap(),
    );
    assert_eq!(
        source.source_diagnostics().unwrap(),
        before,
        "roomy logical cache cannot silently accumulate old windows"
    );
    assert!(manager.evict(&id("u0"), MemoryTier::Device).unwrap());
    let next = manager.acquire(&id("u2"), MemoryTier::Device).unwrap();
    assert_eq!(
        next.device_value("weight")
            .unwrap()
            .evaluated()
            .unwrap()
            .try_as_slice::<i32>()
            .unwrap(),
        [1, 2]
    );
    drop(next);
    receipt.validate().unwrap();
}

#[test]
fn alias_owner_closure_is_persistent_even_when_owner_is_also_requested() {
    let (_directory, source, manager) = super::super::tests::fixture(true, DeviceType::Gpu);
    let plans = manager
        .prepare_disk_reads(&[id("owner"), id("alias")])
        .unwrap();
    let workspace = manager.disk_copy_workspace(&plans, facts()).unwrap();
    assert_eq!(workspace.persistent_units(), [id("owner")]);
    let receipt = workspace
        .receipt(vec![vec![id("owner")], vec![id("alias")]])
        .unwrap();
    let _guard = receipt.activate().unwrap();
    manager.set_admitted_disk_window(&[id("alias")]).unwrap();
    assert_eq!(manager.admitted_disk_persistent_units(), [id("owner")]);
    let lease = manager.acquire(&id("alias"), MemoryTier::Device).unwrap();
    let owner = manager.acquire(&id("owner"), MemoryTier::Device).unwrap();
    for (value, expected) in [
        (lease.device_value("shared").unwrap(), [1, 2]),
        (lease.device_value("local").unwrap(), [3, 4]),
        (owner.device_value("same").unwrap(), [1, 2]),
    ] {
        assert_eq!(
            value.evaluated().unwrap().try_as_slice::<i32>().unwrap(),
            expected
        );
    }
    assert_eq!(
        lease
            .device_value("shared")
            .unwrap()
            .allocation_info()
            .unwrap(),
        owner
            .device_value("weight")
            .unwrap()
            .allocation_info()
            .unwrap()
    );
    assert!(source.source_diagnostics().unwrap().physical_reads > 0);
}

fn replace(manager: &ResidencyManager, value: Array) {
    manager
        .inner
        .state
        .lock()
        .unwrap()
        .storage
        .get_mut(&id("u0"))
        .unwrap()
        .device = Some(
        crate::backend::runtime::residency::manager::ResidentArrays::ordinary(BTreeMap::from([(
            "weight".into(),
            value,
        )])),
    );
}

#[test]
fn stable_envelope_accepts_new_identity_but_rejects_larger_backing_before_ready_hit() {
    let (_directory, source, manager) = fixture();
    drop(manager.acquire(&id("u0"), MemoryTier::Device).unwrap());
    let cpu = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let parent = Array::from_slice(&[1_i32; 128], &[128]);
    let view = parent.try_index_device(..2, &cpu).unwrap();
    view.evaluated().unwrap();
    replace(&manager, view);
    let receipt = receipt(&manager);
    let _guard = receipt.activate().unwrap();
    manager
        .set_admitted_disk_window(&[id("u0"), id("u1")])
        .unwrap();
    replace(&manager, Array::from_slice(&[5_i32, 7], &[2]));
    receipt.validate().unwrap();
    let lease = manager.acquire(&id("u0"), MemoryTier::Device).unwrap();
    assert_eq!(
        lease
            .device_value("weight")
            .unwrap()
            .evaluated()
            .unwrap()
            .try_as_slice::<i32>()
            .unwrap(),
        [5, 7]
    );
    drop(lease);
    let larger = Array::from_slice(&[9_i32; 1024], &[1024]);
    let view = larger.try_index_device(..2, &cpu).unwrap();
    view.evaluated().unwrap();
    replace(&manager, view);
    let before = source.source_diagnostics().unwrap();
    assert!(receipt.validate().is_err());
    assert!(manager.acquire(&id("u0"), MemoryTier::Device).is_err());
    assert_eq!(source.source_diagnostics().unwrap(), before);
}

struct Retired(Arc<AtomicUsize>);
impl Drop for Retired {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
#[test]
fn receipt_clones_do_not_extend_activation_and_guard_drop_takes_no_manager_lock() {
    let (_directory, _source, manager) = fixture();
    let receipt = receipt(&manager);
    let clone = receipt.clone();
    let guard = receipt.activate().unwrap();
    assert!(matches!(
        clone.activate(),
        Err(DiskCopyWorkspaceError::Unproved {
            source: WorkingMemoryError::ExecutionFenced,
            ..
        })
    ));
    let retired = Arc::new(AtomicUsize::new(0));
    drop(crate::backend::ordinary_retirement::OrdinaryRetirement::new(Retired(retired.clone())));
    let state = manager.inner.state.lock().unwrap();
    drop(guard); // Would deadlock if guard destruction tried manager cleanup.
    assert!(state.admitted_disk_route.upgrade().is_none());
    drop(state);
    assert_eq!(
        retired.load(Ordering::SeqCst),
        0,
        "guard drop must not enter native housekeeping"
    );
    assert!(!manager.admitted_disk_route_active());
    let next = clone.activate().unwrap();
    drop(next);
    crate::backend::ordinary_retirement::reclaim();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

fn inject_host(manager: &ResidencyManager, unit: &OffloadUnitId) {
    use crate::backend::runtime::residency::manager::ResidentHostBuffers;
    let mut buffer =
        safemlx::HostTransferBuffer::new(&[2], Dtype::Int32, safemlx::HostTransferPolicy::Transfer)
            .unwrap();
    let values = [13_i32, 17]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    buffer.as_bytes_mut().unwrap().copy_from_slice(&values);
    manager
        .inner
        .state
        .lock()
        .unwrap()
        .storage
        .get_mut(unit)
        .unwrap()
        .host = Some(
        Arc::new(ResidentHostBuffers {
            buffers: [("weight".into(), Arc::new(buffer.freeze()).into())]
                .into_iter()
                .collect(),
        })
        .into(),
    );
}

#[test]
fn next_window_rejects_changed_host_and_geometry_before_changing_allowed_units() {
    let (_directory, source, manager) = fixture();
    drop(manager.acquire(&id("u2"), MemoryTier::Device).unwrap());
    let receipt = receipt(&manager);
    let _guard = receipt.activate().unwrap();
    manager
        .set_admitted_disk_window(&[id("u0"), id("u1")])
        .unwrap();
    let original_window = manager
        .inner
        .state
        .lock()
        .unwrap()
        .admitted_disk_window
        .clone();
    inject_host(&manager, &id("u2"));
    let before = source.source_diagnostics().unwrap();
    assert_fenced(
        manager
            .set_admitted_disk_window(&[id("u1"), id("u2")])
            .unwrap_err(),
    );
    assert_eq!(
        manager.inner.state.lock().unwrap().admitted_disk_window,
        original_window
    );
    assert_eq!(source.source_diagnostics().unwrap(), before);
    let removed = manager
        .inner
        .state
        .lock()
        .unwrap()
        .storage
        .get_mut(&id("u2"))
        .unwrap()
        .host
        .take();
    drop(removed);
    let wrong_dtype = Array::from_slice(&[5_f32, 7.0], &[2]);
    manager
        .inner
        .state
        .lock()
        .unwrap()
        .storage
        .get_mut(&id("u2"))
        .unwrap()
        .device = Some(
        crate::backend::runtime::residency::manager::ResidentArrays::ordinary(BTreeMap::from([(
            "weight".into(),
            wrong_dtype,
        )])),
    );
    let retired = Arc::new(AtomicUsize::new(0));
    drop(crate::backend::ordinary_retirement::OrdinaryRetirement::new(Retired(retired.clone())));
    let error = manager
        .set_admitted_disk_window(&[id("u1"), id("u2")])
        .unwrap_err();
    let ResidencyError::AdmittedDiskRoute { source: error } = error else {
        panic!("unexpected window error");
    };
    assert!(matches!(
        error.downcast_ref::<DiskCopyWorkspaceError>(),
        Some(DiskCopyWorkspaceError::Unproved {
            source: WorkingMemoryError::IdentityMismatch,
            ..
        })
    ));
    assert_eq!(
        manager.inner.state.lock().unwrap().admitted_disk_window,
        original_window
    );
    assert_eq!(source.source_diagnostics().unwrap(), before);
    assert_eq!(
        retired.load(Ordering::SeqCst),
        0,
        "geometry preflight must not run native housekeeping"
    );
    crate::backend::ordinary_retirement::reclaim();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn next_window_preflights_persistent_owner_outside_its_explicit_units() {
    let (_directory, source, manager) = super::super::tests::fixture(true, DeviceType::Gpu);
    let plans = manager
        .prepare_disk_reads(&[id("owner"), id("alias")])
        .unwrap();
    let workspace = manager.disk_copy_workspace(&plans, facts()).unwrap();
    let receipt = workspace
        .receipt(vec![vec![id("owner")], vec![id("alias")]])
        .unwrap();
    let _guard = receipt.activate().unwrap();
    manager.set_admitted_disk_window(&[id("owner")]).unwrap();
    let original_window = manager
        .inner
        .state
        .lock()
        .unwrap()
        .admitted_disk_window
        .clone();
    inject_host(&manager, &id("owner"));
    let before = source.source_diagnostics().unwrap();
    assert_fenced(
        manager
            .set_admitted_disk_window(&[id("alias")])
            .unwrap_err(),
    );
    assert_eq!(
        manager.inner.state.lock().unwrap().admitted_disk_window,
        original_window
    );
    assert_eq!(source.source_diagnostics().unwrap(), before);
}

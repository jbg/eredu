use super::*;
use crate::backend::runtime::residency::storage::RetainedStorageInspectionError;

#[test]
fn immutable_weight_binding_ceiling_survives_actual_host_device_population() {
    let (_dir, store) = fixture_store();
    let manager = manager(
        store,
        OffloadConfig::new(Some(16), Some(16), 1).unwrap(),
        [
            spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            spec("b", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
        ],
        [single("a", "a"), single("b", "b")],
    );
    let bound = manager.prepare_owner_slot_bounds().unwrap().unwrap();
    assert!(std::ptr::eq(bound.source(), &manager));
    assert!(bound.matches(&manager.clone()));
    assert_eq!(
        (
            bound.array_slots(),
            bound.host_slots(),
            bound.source_slots()
        ),
        (2, 2, 0)
    );
    manager.initialize().unwrap();
    manager.prefetch(&id("a"), MemoryTier::Host).unwrap();
    manager.prefetch(&id("a"), MemoryTier::Device).unwrap();
    let after = manager.prepare_owner_slot_bounds().unwrap().unwrap();
    assert_eq!(
        (
            after.array_slots(),
            after.host_slots(),
            after.source_slots()
        ),
        (2, 2, 0)
    );
    let mut arrays = 0;
    let mut hosts = 0;
    assert!(manager
        .try_visit_retained_storage(&mut |owner| -> Result<(), RetainedStorageInspectionError> {
            match owner {
                crate::backend::runtime::residency::storage::RetainedStorageRef::Array(_) => {
                    arrays += 1
                }
                crate::backend::runtime::residency::storage::RetainedStorageRef::Host(_)
                | crate::backend::runtime::residency::storage::RetainedStorageRef::RetainedHost(
                    _,
                ) => hosts += 1,
                _ => {}
            }
            Ok(())
        })
        .unwrap());
    assert_eq!((arrays, hosts), (1, 1));
}

#[test]
fn weight_bound_rejects_locked_state_and_is_bound_to_exact_manager() {
    let (_dir, store) = fixture_store();
    let a = manager(
        store.clone(),
        OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        [single("a", "a")],
    );
    let b = manager(
        store,
        OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
        [spec("a", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk)],
        [single("a", "a")],
    );
    let bound = a.prepare_owner_slot_bounds().unwrap().unwrap();
    assert!(!bound.matches(&b));
    let lock = a.inner.state.lock().unwrap();
    assert!(matches!(
        a.prepare_owner_slot_bounds(),
        Err(RetainedStorageInspectionError::Busy)
    ));
    drop(lock);
    assert!(a.prepare_owner_slot_bounds().unwrap().is_some());
}

#[test]
fn ordinary_window_trim_preserves_canonical_alias_pins_and_rejects_other_live_leases() {
    let (_dir, store) = fixture_store();
    let owner = WeightBinding::new("weight", "a", TensorSelection::Full, 8)
        .unwrap()
        .with_logical_target("canonical")
        .unwrap();
    let alias = WeightBinding::alias("weight", "canonical", 8).unwrap();
    let manager = manager(
        store,
        OffloadConfig::new(None, None, 1).unwrap(),
        [
            spec("owner", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            spec("alias", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
            spec("leased", 8, ResidencyPolicy::Cacheable, MemoryTier::Disk),
        ],
        [
            unit("owner", [owner]),
            unit(
                "alias",
                [binding("own", "b", TensorSelection::Full, 8), alias],
            ),
            single("leased", "c"),
        ],
    );
    manager.initialize().unwrap();
    manager.prefetch(&id("alias"), MemoryTier::Device).unwrap();
    assert!(!manager.admitted_disk_route_active());
    manager
        .trim_device_units(&[id("owner"), id("alias")], &[id("alias")])
        .unwrap();
    let alias = manager.acquire(&id("alias"), MemoryTier::Device).unwrap();
    assert_eq!(
        alias
            .device_value("weight")
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<i32>(),
        &[1, 2]
    );
    let leased = manager.acquire(&id("leased"), MemoryTier::Device).unwrap();
    assert!(matches!(
        manager.trim_device_units(&[id("leased")], &[]),
        Err(ResidencyError::Ledger(
            ResidencyLedgerError::InUseEviction { pin_count: 1, .. }
        ))
    ));
    drop(leased);
    manager.trim_device_units(&[id("leased")], &[]).unwrap();
    assert!(manager
        .inner
        .state
        .lock()
        .unwrap()
        .alias_owner_pins
        .contains(&id("owner"), MemoryTier::Device));
}

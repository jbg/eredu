//! Hardware discovery preserves the separately admitted native device birth.
#![cfg(all(feature = "metal", target_vendor = "apple"))]

#[test]
fn topology_discovery_preserves_prepared_metal_device_provenance() {
    let devices = safemlx::physical_memory_topology().unwrap();
    assert_eq!(devices.len(), 1);
    assert_eq!(safemlx::physical_memory_topology().unwrap(), devices);
    assert_eq!(
        safemlx::default_allocation_placement().unwrap(),
        safemlx::AllocationPlacement::Host
    );
    assert_eq!(
        safemlx::gpu_allocation_placement().unwrap(),
        safemlx::AllocationPlacement::Host
    );
    let prepared = safemlx::PreparedMetalDevice::try_new(()).unwrap();
    let initialized = prepared.try_initialize().unwrap();
    initialized.try_borrow().unwrap();
    assert_eq!(safemlx::physical_memory_topology().unwrap(), devices);
}

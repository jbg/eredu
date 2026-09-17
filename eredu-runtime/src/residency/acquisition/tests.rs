use super::*;
use eredu_core::residency::{ResidencyAdmissionStorage, ResidencyReservationRow};
fn controller() -> (ResidencyController, [OffloadUnitId; 3]) {
    let ids = ["a", "b", "c"].map(|name| OffloadUnitId::new(name).unwrap());
    let catalog = Catalog(
        ["a", "b", "c"]
            .map(|key| (key.into(), metadata(key, vec![1])))
            .into(),
    );
    let units = ids.iter().map(|id| {
        OffloadUnit::new(
            id.clone(),
            [WeightBinding::new("weight", id.as_str(), TensorSelection::Full, 4).unwrap()],
        )
        .unwrap()
    });
    let plan = OffloadPlan::new(
        OffloadConfig::new(Some(8), Some(8), 1).unwrap(),
        ids.iter().map(|id| {
            OffloadUnitSpec::new(id.clone(), 4, ResidencyPolicy::Cacheable, MemoryTier::Disk)
                .unwrap()
        }),
    )
    .unwrap();
    (
        ResidencyController::new(&catalog, plan, units).unwrap(),
        ids,
    )
}
#[test]
fn independent_snapshot_survives_reservation_and_partial_publication_without_reclassifying_hits() {
    let (mut controller, ids) = controller();
    controller.ledger_mut().mark_initialized();
    controller
        .ledger_mut()
        .reserve_copy(&ids[0], MemoryTier::Device, 4, &BTreeSet::new())
        .unwrap();
    controller
        .ledger_mut()
        .publish_reserved(&ids[0], MemoryTier::Device, 4, None)
        .unwrap();
    let mut flags = [false; 2];
    let mut order = [0; 2];
    let before = flags.as_ptr();
    let snapshot = controller
        .plan_acquisition_in(&ids[..2], MemoryTier::Device, false, &mut order, &mut flags)
        .unwrap();
    assert_eq!(snapshot.missing(), [false, true]);
    assert_eq!(snapshot.missing().as_ptr(), before);
    assert!(std::ptr::eq(snapshot.ids().as_ptr(), ids.as_ptr()));
    let source = controller.ledger().plan_source();
    let storage = ResidencyAdmissionStorage::try_new(source, 2, 1).unwrap();
    let storage = controller
        .reserve_acquisition_in(
            snapshot,
            &[ResidencyReservationRow { input: 1, bytes: 4 }],
            MemoryTier::Device,
            storage,
        )
        .unwrap();
    assert!(storage.evicted().is_empty());
    controller
        .ledger_mut()
        .publish_reserved(&ids[1], MemoryTier::Device, 3, None)
        .unwrap();
    controller
        .touch_acquisition_hits_ref(snapshot, MemoryTier::Device)
        .unwrap();
    controller
        .rollback_acquisition_ref(snapshot, MemoryTier::Device)
        .unwrap();
    assert_eq!(
        snapshot.missing(),
        [false, true],
        "publication must not change the original snapshot"
    );
    assert!(controller
        .ledger()
        .is_resident(&ids[0], MemoryTier::Device)
        .unwrap());
    assert!(controller
        .ledger()
        .is_resident(&ids[1], MemoryTier::Device)
        .unwrap());
    assert_eq!(
        controller
            .ledger()
            .telemetry()
            .resident_bytes()
            .get(MemoryTier::Device),
        7
    );
}
#[test]
fn snapshot_refusals_preserve_destination_and_existing_initialization_precedence() {
    let (mut controller, ids) = controller();
    let mut flags = [true, false];
    let mut order = [0; 2];
    assert!(matches!(
        controller.plan_acquisition_in(&ids[..2], MemoryTier::Disk, false, &mut order, &mut flags),
        Err(ResidencyAcquisitionFailure::NotInitialized)
    ));
    assert_eq!(flags, [true, false]);
    controller.ledger_mut().mark_initialized();
    assert!(matches!(
        controller.plan_acquisition_in(&ids, MemoryTier::Device, false, &mut [0; 3], &mut flags),
        Err(ResidencyAcquisitionFailure::Destination {
            required: 3,
            available: 2
        })
    ));
    assert_eq!(flags, [true, false]);
    let duplicate = [ids[0].clone(), ids[0].clone()];
    let actual = controller
        .plan_acquisition_in(
            &duplicate,
            MemoryTier::Device,
            false,
            &mut order,
            &mut flags,
        )
        .unwrap_err()
        .into_owned();
    assert_eq!(actual, ResidencyLedgerError::DuplicateBatchUnit);
    assert_eq!(flags, [true, false]);
}

use super::*;
use crate::backend::runtime::residency::manager::ForegroundDiskSourceCapacity;

#[test]
fn foreground_source_capacity_refuses_another_request_in_the_same_pool() {
    let mut prepared = OriginalOperationFixture::prepare(true);
    // Add the actual budget control and permitted backing at the fixture's
    // genuine pre-seal control query. No buffer or source payload is created.
    let bytes = ForegroundDiskSourceCapacity::control_bytes()
        .unwrap()
        .checked_add(8)
        .unwrap();
    POINTWISE_CONTROLS.with(|slot| assert!(slot.replace(Some(bytes as u64)).is_none()));
    let reset = PointwiseControlsReset;
    let (old, old_guard, pool) = with_prepared_original_operation_reservation(
        None,
        &mut prepared,
        |controls, _observer, pool, bank, reservation| {
            assert!(bank.is_none());
            let capacity =
                ForegroundDiskSourceCapacity::new(8, controls.clone(), reservation).unwrap();
            capacity.validate_reservation(reservation).unwrap();
            (capacity, controls.clone(), pool.clone())
        },
    );
    with_prepared_original_operation_reservation(
        None,
        &mut prepared,
        |controls, _observer, current_pool, bank, reservation| {
            assert!(bank.is_none());
            assert!(current_pool.same_domain(&pool));
            assert!(matches!(
                old.validate_reservation(reservation),
                Err(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch)
            ));
            assert!(matches!(
                ForegroundDiskSourceCapacity::new(8, old_guard.clone(), reservation),
                Err(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch)
            ));
            let current =
                ForegroundDiskSourceCapacity::new(8, controls.clone(), reservation).unwrap();
            current.validate_reservation(reservation).unwrap();
            // Refused cross-request adoption cannot spend either budget. These
            // independent, full-size permits need no native construction.
            let old_full = old.try_acquire(8).unwrap();
            let current_full = current.try_acquire(8).unwrap();
            assert!(old.try_acquire(1).is_err());
            assert!(current.try_acquire(1).is_err());
            drop((old_full, current_full, current));
        },
    );
    drop((old, old_guard, pool, reset));
    prepared.finish();
}

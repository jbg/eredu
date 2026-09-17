use super::*;
#[test]
fn replacement_publication_keeps_both_backings_until_their_actual_owners_retire() {
    let pool = CacheResidencyPool::new(CachePoolLimits::new(32, 64, 64, 0).unwrap());
    let member = pool.register_manager(7).unwrap();
    pool.update_manager(
        7,
        CachePoolUsage {
            host_bytes: 48,
            ..CachePoolUsage::default()
        },
    )
    .unwrap();
    let mut destination = pool
        .reserve(CachePoolUsage {
            device_bytes: 24,
            transfer_in_flight_bytes: 48,
            ..CachePoolUsage::default()
        })
        .unwrap();
    let before = pool.report().unwrap();
    let foreign_pool = CacheResidencyPool::new(pool.limits());
    let foreign = foreign_pool.register_manager(7).unwrap();
    assert_eq!(
        destination
            .publish_retaining_replaced_storage(&foreign, CachePoolUsage::default())
            .unwrap_err(),
        CachePoolError::ForeignMembership
    );
    assert!(
        destination
            .publish_retaining_replaced_storage(
                &member,
                CachePoolUsage {
                    device_bytes: 25,
                    ..CachePoolUsage::default()
                }
            )
            .is_err()
    );
    assert!(
        destination
            .publish_to_manager(
                &member,
                CachePoolUsage {
                    device_bytes: 24,
                    ..CachePoolUsage::default()
                }
            )
            .is_err()
    );
    assert_eq!(pool.report().unwrap(), before);
    destination
        .publish_retaining_replaced_storage(
            &member,
            CachePoolUsage {
                device_bytes: 24,
                ..CachePoolUsage::default()
            },
        )
        .unwrap();
    assert_eq!(pool.report().unwrap(), before);
    // Canonical device state can retire first; the old host source/transfer
    // still belongs to the exact reservation, without another admission.
    drop(member);
    let retained = pool.report().unwrap();
    assert_eq!(retained.current_device_bytes, 0);
    assert_eq!(retained.current_host_bytes, 48);
    assert_eq!(retained.current_transfer_in_flight_bytes, 48);
    drop(destination);
    let retired = pool.report().unwrap();
    assert_eq!(retired.current_host_bytes, 0);
    assert_eq!(retired.current_transfer_in_flight_bytes, 0);
}

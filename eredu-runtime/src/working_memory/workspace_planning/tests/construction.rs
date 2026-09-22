use super::*;
use eredu_core::{DomainMemoryRequirements, MemoryLimit, MemoryLimits};

#[test]
fn constructor_metadata_uses_host_charge_and_preserves_live_ceilings() {
    for configured in [MemoryLimit::Finite(1 << 20), MemoryLimit::Unlimited] {
        let fixture = crate::working_memory::memory_fixture::separate::device_ledger(0, 0).unwrap();
        let topology = fixture.topology_handle();
        let host = topology.host_domain();
        let device = crate::working_memory::memory_fixture::separate::device_domain(&fixture);
        let limits = MemoryLimits::resolve(
            &topology,
            [(host, configured), (device, MemoryLimit::Finite(0))],
        )
        .unwrap();
        let baseline = DomainMemoryRequirements::zero(&topology);
        let pool = MemoryLedger::new(topology, limits, baseline).unwrap();
        let initial = pool.snapshot().unwrap();
        let execution = InferenceExecutionIdentity::default();
        let ceiling =
            MemoryLimits::resolve(pool.topology(), [(host, MemoryLimit::Finite(1 << 19))]).unwrap();
        let predecessor = pool
            .prepare_workspace_metadata(&execution, ceiling)
            .unwrap();
        let before = pool.snapshot().unwrap();
        let owner = pool.prepare_construction_metadata().unwrap();
        owner.funding().reserve_metadata(128).unwrap();
        let after = pool.snapshot().unwrap();
        assert!(after.domains[0].current_charge_bytes > before.domains[0].current_charge_bytes);
        assert_eq!(
            after.domains[0].effective_limit,
            MemoryLimit::Finite(1 << 19)
        );
        assert_eq!(after.domains[1], before.domains[1]);
        assert!(pool.acquire_unquoted().is_err());

        let available = (1 << 19) - after.domains[0].current_charge_bytes;
        owner
            .funding()
            .reserve_metadata(usize::try_from(available).unwrap())
            .unwrap();
        let full = pool.snapshot().unwrap();
        let next_id = pool.0.usage.lock().unwrap().next_funding;
        assert!(matches!(
            owner.funding().reserve_metadata(1),
            Err(HostMetadataFundingError::Domain(
                eredu_core::MemoryDomainError::BudgetExceeded { domain, requested_bytes: 1, .. }
            )) if domain == host
        ));
        assert!(matches!(
            pool.prepare_construction_metadata(),
            Err(HostMetadataFundingError::Domain(
                eredu_core::MemoryDomainError::BudgetExceeded { domain, .. }
            )) if domain == host
        ));
        assert_eq!(pool.snapshot().unwrap(), full);
        assert_eq!(pool.0.usage.lock().unwrap().next_funding, next_id);

        let last = owner.funding().clone();
        let owner = owner.seal().unwrap();
        let mut full = full;
        full.reservations -= 1;
        assert!(pool.acquire_unquoted().is_err());
        drop(owner);
        assert_eq!(pool.snapshot().unwrap(), full);
        drop(last);
        assert_eq!(
            pool.snapshot().unwrap().domains[0].current_charge_bytes,
            before.domains[0].current_charge_bytes
        );
        drop(predecessor);
        let retired = pool.snapshot().unwrap();
        assert_eq!(
            retired.domains[0].current_charge_bytes,
            initial.domains[0].current_charge_bytes
        );
        assert_eq!(retired.domains[0].effective_limit, configured);
        assert_eq!(retired.reservations, initial.reservations);
    }
}

#[test]
fn unlimited_constructor_metadata_preserves_overflow_and_unquoted_exclusion() {
    let topology = crate::working_memory::memory_fixture::host_topology();
    let limits = MemoryLimits::unlimited(&topology);
    let baseline = DomainMemoryRequirements::zero(&topology);
    let pool = MemoryLedger::new(topology, limits, baseline).unwrap();
    let unquoted = pool.acquire_unquoted().unwrap();
    let before = pool.snapshot().unwrap();
    let next_id = pool.0.usage.lock().unwrap().next_funding;
    assert!(matches!(
        pool.prepare_construction_metadata(),
        Err(HostMetadataFundingError::Unavailable)
    ));
    assert_eq!(pool.snapshot().unwrap(), before);
    assert_eq!(pool.0.usage.lock().unwrap().next_funding, next_id);
    drop(unquoted);
    let owner = pool.prepare_construction_metadata().unwrap();
    let before = pool.snapshot().unwrap();
    assert!(owner.funding().reserve_metadata(usize::MAX).is_err());
    assert_eq!(pool.snapshot().unwrap(), before);
    drop(owner);
}

#[test]
fn sealed_cold_metadata_keeps_its_charge_and_closes_every_growth_alias() {
    for separate in [false, true] {
        for configured in [MemoryLimit::Finite(1 << 20), MemoryLimit::Unlimited] {
            let topology = if separate {
                crate::working_memory::memory_fixture::separate::device_ledger(0, 0)
                    .unwrap()
                    .topology_handle()
            } else {
                crate::working_memory::memory_fixture::host_topology()
            };
            let limits = MemoryLimits::resolve(
                &topology,
                topology.domains().map(|(domain, _)| {
                    (
                        domain,
                        if domain == topology.host_domain() {
                            configured
                        } else {
                            MemoryLimit::Finite(0)
                        },
                    )
                }),
            )
            .unwrap();
            let baseline = DomainMemoryRequirements::zero(&topology);
            let pool = MemoryLedger::new(topology, limits, baseline).unwrap();
            let initial = pool.snapshot().unwrap();
            let producer = pool.prepare_construction_metadata().unwrap();
            producer.funding().reserve_metadata(128).unwrap();
            let payload = vec![7u8; 128];
            let alias = producer.funding().clone();
            let id = producer.id;
            let mut expected = pool.snapshot().unwrap();
            let next_id = pool.0.usage.lock().unwrap().next_funding;
            assert_eq!(expected.reservations, 1);
            assert!(pool.acquire_unquoted().is_err());

            let retained = producer.seal().unwrap();
            expected.reservations = 0;
            assert_eq!(pool.snapshot().unwrap(), expected);
            assert_eq!(pool.0.usage.lock().unwrap().next_funding, next_id);
            for bytes in [0, 1, usize::MAX] {
                assert_eq!(
                    alias.reserve_metadata(bytes),
                    Err(HostMetadataFundingError::Unavailable)
                );
            }
            assert_eq!(pool.snapshot().unwrap(), expected);
            assert!(
                pool.0
                    .usage
                    .lock()
                    .unwrap()
                    .funding
                    .grow_planning(id, 1)
                    .is_err()
            );
            assert_eq!(pool.snapshot().unwrap(), expected);
            let unquoted = pool.acquire_unquoted().unwrap();
            let during = pool.snapshot().unwrap();
            for (during, expected) in during.domains.iter().zip(&expected.domains) {
                assert_eq!(
                    during.outstanding_reservation_bytes,
                    expected.outstanding_reservation_bytes
                );
                assert_eq!(during.effective_limit, expected.effective_limit);
            }
            drop(unquoted);
            for (expected, during) in expected.domains.iter_mut().zip(&during.domains) {
                expected.historical_peak_bytes = during.historical_peak_bytes;
            }
            assert_eq!(pool.snapshot().unwrap(), expected);
            drop(retained);
            assert_eq!(pool.snapshot().unwrap(), expected);
            assert!(payload.iter().all(|value| *value == 7));
            drop(payload);
            drop(alias);
            let retired = pool.snapshot().unwrap();
            assert_eq!(retired.reservations, initial.reservations);
            for (retired, initial) in retired.domains.iter().zip(&initial.domains) {
                assert_eq!(retired.current_charge_bytes, initial.current_charge_bytes);
                assert_eq!(retired.effective_limit, initial.effective_limit);
            }
        }
    }
}

#[test]
fn construction_seal_rejects_foreign_stale_and_request_authority_atomically() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 20, 0).unwrap();
    let producer = pool.prepare_construction_metadata().unwrap();
    let execution = InferenceExecutionIdentity::default();
    let request = pool
        .prepare_workspace_metadata(&execution, MemoryLimits::unlimited(pool.topology()))
        .unwrap();
    let before = pool.snapshot().unwrap();
    let mut usage = pool.0.usage.lock().unwrap();
    let next_id = usage.next_funding;
    assert!(
        usage
            .funding
            .seal_construction_metadata(producer.id, &execution)
            .is_err()
    );
    assert!(
        usage
            .funding
            .seal_construction_metadata(next_id, pool.construction_identity())
            .is_err()
    );
    let request_id = next_id - 1;
    assert!(
        usage
            .funding
            .seal_construction_metadata(request_id, pool.construction_identity())
            .is_err()
    );
    assert_eq!(usage.next_funding, next_id);
    drop(usage);
    assert_eq!(pool.snapshot().unwrap(), before);
    producer.funding().reserve_metadata(1).unwrap();
    let retained = producer.seal().unwrap();
    assert!(pool.acquire_unquoted().is_err());
    drop(request);
    assert!(pool.acquire_unquoted().is_ok());
    drop(retained);
    assert_eq!(pool.snapshot().unwrap().reservations, 0);
}

#[test]
fn poisoned_construction_seal_preserves_charge_and_exclusion() {
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 20, 0).unwrap();
    let producer = pool.prepare_construction_metadata().unwrap();
    producer.funding().reserve_metadata(128).unwrap();
    let alias = producer.funding().clone();
    let before = {
        let usage = pool.0.usage.lock().unwrap();
        (usage.reserved, usage.reservations, usage.next_funding)
    };
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _usage = pool.0.usage.lock().unwrap();
        panic!("poison construction coordinator");
    }));
    assert!(unwind.is_err());
    assert!(matches!(
        producer.seal(),
        Err(HostMetadataFundingError::Poisoned)
    ));
    drop(alias);
    let usage = pool.0.usage.lock().unwrap_err().into_inner();
    assert_eq!(
        (usage.reserved, usage.reservations, usage.next_funding),
        before
    );
}

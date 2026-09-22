//! Physical-domain behavior through the public coordinator operations.
use eredu_core::*;
use eredu_runtime::working_memory::*;
use std::{
    num::NonZeroU8,
    sync::{Arc, Barrier},
};

fn gpu(n: u32) -> MemoryLocation {
    MemoryLocation::Device(MemoryDeviceId {
        backend: "conformance",
        ordinal: n,
    })
}
fn topology(unified: bool) -> Arc<MemoryTopology> {
    Arc::new(
        MemoryTopology::new(if unified {
            vec![MemoryDomainDescription {
                name: "shared".into(),
                locations: vec![MemoryLocation::Host, gpu(0), gpu(1)],
            }]
        } else {
            vec![
                MemoryDomainDescription {
                    name: "host".into(),
                    locations: vec![MemoryLocation::Host],
                },
                MemoryDomainDescription {
                    name: "gpu0".into(),
                    locations: vec![gpu(0)],
                },
                MemoryDomainDescription {
                    name: "gpu1".into(),
                    locations: vec![gpu(1)],
                },
            ]
        })
        .unwrap(),
    )
}
fn ledger(unified: bool, limits: [MemoryLimit; 3]) -> MemoryLedger {
    let topology = topology(unified);
    let limits = MemoryLimits::resolve(
        &topology,
        topology
            .domains()
            .enumerate()
            .map(|(i, (d, _))| (d, limits[i])),
    )
    .unwrap();
    let baseline = DomainMemoryRequirements::zero(&topology);
    MemoryLedger::new(topology, limits, baseline).unwrap()
}
fn allocation(pool: &MemoryLedger, location: MemoryLocation, bytes: u64) -> StorageAllocation {
    StorageAllocation::new(
        bytes,
        Arc::new(
            MemoryPlacement::fixed(
                pool.topology(),
                pool.topology().domain_for(location).unwrap(),
            )
            .unwrap(),
        ),
    )
}
fn request(pool: &MemoryLedger, charges: &[(MemoryLocation, u64)]) -> Admission {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 1,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let zero = || DomainMemoryRequirements::zero(pool.topology());
    let mut requirements = zero();
    for &(location, bytes) in charges {
        requirements
            .add_allocation(bytes, allocation(pool, location, bytes).placement())
            .unwrap();
    }
    let empty = || WorkspaceBound::bounded(0, "no allocation in this fixture component");
    let diagnostic = charges
        .iter()
        .try_fold(0u64, |sum, (_, bytes)| sum.checked_add(*bytes));
    Admission {
        requested_positions: 2,
        incremental_required_bytes: diagnostic,
        additional_headroom: MemoryHeadroomDeclarations::default(),
        memory_limits: MemoryLimitDeclarations::default(),
        state: RuntimeStateEstimate {
            fixed_state_bytes: 0,
            bytes_per_position_per_batch: 0,
            context_state_bytes: 0,
            selected_state_backing: None,
            multimodal_embedding_bytes: 0,
            media_execution_workspace_bytes: 0,
            requested_state_bytes: 0,
            persistent_state_completeness: EstimationCompleteness::Complete,
            completeness: EstimationCompleteness::Complete,
            assumptions: StateMemoryAssumptions {
                floating_state_dtype_bytes: NonZeroU8::new(4).unwrap(),
                batch_size: 1,
                requested_positions: 2,
                sliding_window_bounds: vec![],
                allocation_granularity: 1,
            },
            physical_domains: Some(DomainRuntimeStateEstimate {
                geometry,
                decoder_state: zero(),
                media_embeddings: zero(),
                media_workspace: zero(),
            }),
            execution_workspace: Some(ExecutionWorkspaceEstimate {
                geometry,
                activations: diagnostic.map_or_else(
                    || WorkspaceBound::PerDomain {
                        assumptions: "independent fixture domain charges exceed one aggregate"
                            .into(),
                    },
                    |bytes| WorkspaceBound::bounded(bytes, "fixture allocation population"),
                ),
                attention: empty(),
                vocabulary: empty(),
                state_update: empty(),
                materialization: empty(),
                retained: empty(),
                physical_domains: Some(DomainExecutionWorkspaceEstimate {
                    geometry,
                    activations: requirements,
                    attention: zero(),
                    vocabulary: zero(),
                    state_update: zero(),
                    materialization: zero(),
                    retained: zero(),
                }),
            }),
        },
    }
}
fn allocation_delta(
    pool: &MemoryLedger,
    baseline: &MemoryLedgerSnapshot,
    location: MemoryLocation,
) -> u64 {
    let domain = pool.topology().domain_for(location).unwrap();
    let now = pool.snapshot().unwrap();
    let charge = |s: &MemoryDomainSnapshot| {
        s.current_charge_bytes
            .checked_sub(s.registry_metadata_bytes)
            .unwrap()
            .checked_sub(s.reservation_control_bytes)
            .unwrap()
    };
    charge(now.domains.iter().find(|s| s.domain == domain).unwrap())
        - charge(
            baseline
                .domains
                .iter()
                .find(|s| s.domain == domain)
                .unwrap(),
        )
}

#[test]
fn same_transfer_operations_follow_physical_topology_and_preserve_independent_retirement() {
    for unified in [false, true] {
        for finite in [false, true] {
            let limit = if finite {
                MemoryLimit::Finite(1_000_000)
            } else {
                MemoryLimit::Unlimited
            };
            let pool = ledger(unified, [limit; 3]);
            let baseline = pool.snapshot().unwrap();
            let source = StoragePublicationLayout::new(1)
                .unwrap()
                .fund(&pool)
                .unwrap()
                .register_storage([(1u32, allocation(&pool, MemoryLocation::Host, 64))])
                .unwrap();
            let pin = StoragePublicationLayout::new(1)
                .unwrap()
                .fund(&pool)
                .unwrap()
                .pin_registered_storage([(1u32, 64)])
                .unwrap();
            let mut outputs = StoragePublicationLayout::new(2)
                .unwrap()
                .fund(&pool)
                .unwrap()
                .register_storage_individually([
                    (2u32, allocation(&pool, MemoryLocation::Host, 16)),
                    (3u32, allocation(&pool, gpu(0), 64)),
                ])
                .unwrap();
            assert_eq!(
                allocation_delta(&pool, &baseline, MemoryLocation::Host),
                if unified { 144 } else { 80 }
            );
            assert_eq!(
                allocation_delta(&pool, &baseline, gpu(0)),
                if unified { 144 } else { 64 }
            );
            drop(source);
            drop(outputs.remove(&2));
            assert_eq!(
                allocation_delta(&pool, &baseline, MemoryLocation::Host),
                if unified { 128 } else { 64 }
            );
            drop(pin);
            assert_eq!(
                allocation_delta(&pool, &baseline, MemoryLocation::Host),
                if unified { 64 } else { 0 }
            );
            drop(outputs);
            assert!(pool
                .snapshot()
                .unwrap()
                .domains
                .iter()
                .zip(&baseline.domains)
                .all(|(a, b)| a.current_charge_bytes == b.current_charge_bytes));
        }
    }
}

#[test]
fn last_domain_failure_preserves_all_counters_and_peaks_for_registration_and_reservation() {
    let pool = ledger(
        false,
        [
            MemoryLimit::Unlimited,
            MemoryLimit::Finite(64),
            MemoryLimit::Finite(63),
        ],
    );
    let funding = pool.prepare_storage_metadata().unwrap();
    let prepared = StoragePublicationLayout::new(3)
        .unwrap()
        .prepare(&pool, &funding)
        .unwrap();
    let before = pool.snapshot().unwrap();
    let entries = [
        (1u32, allocation(&pool, MemoryLocation::Host, 16)),
        (2u32, allocation(&pool, gpu(0), 64)),
        (3u32, allocation(&pool, gpu(1), 64)),
    ];
    assert!(matches!(
        prepared.register_storage(entries),
        Err(WorkingMemoryError::Domain(
            MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(pool.snapshot().unwrap(), before);
    let admission = request(
        &pool,
        &[(MemoryLocation::Host, 16), (gpu(0), 64), (gpu(1), 64)],
    );
    assert!(matches!(
        pool.reserve(&InferenceExecutionIdentity::default(), &admission),
        Err(WorkingMemoryError::Domain(
            MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
    assert_eq!(pool.snapshot().unwrap(), before);
    let accepted = pool
        .reserve(
            &InferenceExecutionIdentity::default(),
            &request(&pool, &[(gpu(0), 64), (gpu(1), 63)]),
        )
        .unwrap();
    assert_eq!(allocation_delta(&pool, &before, gpu(0)), 64);
    drop(accepted);
    assert_eq!(allocation_delta(&pool, &before, gpu(0)), 0);
    assert_eq!(allocation_delta(&pool, &before, gpu(1)), 0);
}

#[test]
fn unlimited_still_checks_arithmetic_and_finite_maximum_is_exact() {
    for limit in [MemoryLimit::Unlimited, MemoryLimit::Finite(u64::MAX)] {
        let pool = ledger(false, [MemoryLimit::Unlimited, limit, limit]);
        let left = StoragePublicationLayout::new(1)
            .unwrap()
            .fund(&pool)
            .unwrap()
            .register_storage([(1u32, allocation(&pool, gpu(0), u64::MAX))])
            .unwrap();
        let right = StoragePublicationLayout::new(1)
            .unwrap()
            .fund(&pool)
            .unwrap()
            .register_storage([(2u32, allocation(&pool, gpu(1), u64::MAX))])
            .unwrap();
        let funding = pool.prepare_storage_metadata().unwrap();
        let prepared = StoragePublicationLayout::new(1)
            .unwrap()
            .prepare(&pool, &funding)
            .unwrap();
        let before = pool.snapshot().unwrap();
        assert!(matches!(
            prepared.register_storage([(3u32, allocation(&pool, gpu(0), 1))]),
            Err(WorkingMemoryError::Domain(MemoryDomainError::Overflow))
                | Err(WorkingMemoryError::Overflow)
        ));
        assert_eq!(pool.snapshot().unwrap(), before);
        drop((left, right));
    }
    let pool = ledger(
        false,
        [
            MemoryLimit::Unlimited,
            MemoryLimit::Finite(0),
            MemoryLimit::Unlimited,
        ],
    );
    let zero = StoragePublicationLayout::new(1)
        .unwrap()
        .fund(&pool)
        .unwrap()
        .register_storage([(1u32, allocation(&pool, gpu(0), 0))])
        .unwrap();
    assert!(StoragePublicationLayout::new(1)
        .unwrap()
        .fund(&pool)
        .unwrap()
        .register_storage([(2u32, allocation(&pool, gpu(0), 1))])
        .is_err());
    drop(zero);
}

#[test]
fn managed_allowances_deduplicate_physical_domains_and_survive_aliases() {
    for unified in [false, true] {
        let pool = ledger(unified, [MemoryLimit::Unlimited; 3]);
        let baseline = pool.snapshot().unwrap();
        let placement = MemoryPlacement::possible_locations(
            pool.topology(),
            [MemoryLocation::Host, gpu(0), gpu(1), gpu(0)],
            "fixture managed allocator candidates".into(),
        )
        .unwrap();
        let descriptor = StorageAllocation::new(64, Arc::new(placement));
        let first = StoragePublicationLayout::new(1)
            .unwrap()
            .fund(&pool)
            .unwrap()
            .register_storage([(1u32, descriptor.clone())])
            .unwrap();
        let view = StoragePublicationLayout::new(1)
            .unwrap()
            .fund(&pool)
            .unwrap()
            .register_storage([(1u32, descriptor)])
            .unwrap();
        for domain in pool.snapshot().unwrap().domains {
            assert_eq!(domain.estimated_placement_allowance_bytes, 64);
            assert_eq!(
                domain.placement_allowance_basis,
                Some(PlacementAllowanceBasis::FullCapacityInEveryCandidateDomain)
            );
        }
        assert_eq!(allocation_delta(&pool, &baseline, gpu(0)), 64);
        let funding = pool.prepare_storage_metadata().unwrap();
        let prepared = StoragePublicationLayout::new(1)
            .unwrap()
            .prepare(&pool, &funding)
            .unwrap();
        let before = pool.snapshot().unwrap();
        assert!(matches!(
            prepared.register_storage([(1u32, allocation(&pool, gpu(0), 64))]),
            Err(WorkingMemoryError::StoragePlacementMismatch)
        ));
        assert_eq!(pool.snapshot().unwrap(), before);
        drop(first);
        assert_eq!(allocation_delta(&pool, &baseline, gpu(0)), 64);
        drop(view);
        assert_eq!(allocation_delta(&pool, &baseline, gpu(0)), 0);
    }
}

#[test]
fn competing_reservations_commit_one_complete_domain_vector() {
    let pool = ledger(
        false,
        [
            MemoryLimit::Unlimited,
            MemoryLimit::Finite(64),
            MemoryLimit::Finite(64),
        ],
    );
    let before = pool.snapshot().unwrap();
    let gate = Arc::new(Barrier::new(3));
    let results = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let gate = Arc::clone(&gate);
                let pool = pool.clone();
                scope.spawn(move || {
                    let request = request(
                        &pool,
                        &[(MemoryLocation::Host, 16), (gpu(0), 64), (gpu(1), 64)],
                    );
                    gate.wait();
                    let result = pool.reserve(&InferenceExecutionIdentity::default(), &request);
                    gate.wait();
                    result
                })
            })
            .collect();
        gate.wait();
        gate.wait();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(results.iter().filter(|value| value.is_ok()).count(), 1);
    assert_eq!(allocation_delta(&pool, &before, gpu(0)), 64);
    assert_eq!(allocation_delta(&pool, &before, gpu(1)), 64);
    drop(results);
    assert_eq!(allocation_delta(&pool, &before, gpu(0)), 0);
}

#[test]
fn funded_publication_converts_charges_and_completion_controls_retirement() {
    for unified in [false, true] {
        let pool = ledger(unified, [MemoryLimit::Unlimited; 3]);
        let before = pool.snapshot().unwrap();
        let publication = MemoryLedger::storage_metadata_control_bytes().unwrap()
            + StoragePublicationLayout::<u32>::new(2)
                .unwrap()
                .requested_bytes();
        let (metadata, run) = pool
            .reserve(
                &InferenceExecutionIdentity::default(),
                &request(
                    &pool,
                    &[(MemoryLocation::Host, 16 + publication), (gpu(0), 64)],
                ),
            )
            .unwrap()
            .into_funding()
            .unwrap();
        let reserved = pool.snapshot().unwrap();
        let scope = run.scope().unwrap();
        let mut registrations = StoragePublicationLayout::new(2)
            .unwrap()
            .fund_from(&scope)
            .unwrap()
            .adopt_storage_individually(
                &scope,
                [
                    (1u32, allocation(&pool, MemoryLocation::Host, 16)),
                    (2u32, allocation(&pool, gpu(0), 64)),
                ],
            )
            .unwrap();
        for (before, after) in reserved
            .domains
            .iter()
            .zip(pool.snapshot().unwrap().domains)
        {
            assert_eq!(before.current_charge_bytes, after.current_charge_bytes);
        }
        run.close().unwrap();
        // Exact native certification is what releases unused allowance.
        scope.certify().unwrap();
        drop(registrations.remove(&1));
        assert_eq!(
            allocation_delta(&pool, &before, MemoryLocation::Host),
            if unified { 64 } else { 0 }
        );
        drop(metadata);
        assert_eq!(allocation_delta(&pool, &before, gpu(0)), 64);
        drop(registrations);
        assert_eq!(allocation_delta(&pool, &before, gpu(0)), 0);
        assert_eq!(pool.snapshot().unwrap().funding_accounts, 0);
    }
}

#[test]
fn missing_attribution_rejects_unlimited_and_foreign_topology_never_changes_usage() {
    let pool = ledger(false, [MemoryLimit::Unlimited; 3]);
    let mut admission = request(&pool, &[(gpu(0), 1)]);
    admission.state.physical_domains = None;
    let funding = pool.prepare_storage_metadata().unwrap();
    let prepared = StoragePublicationLayout::new(1)
        .unwrap()
        .prepare(&pool, &funding)
        .unwrap();
    let before = pool.snapshot().unwrap();
    assert!(matches!(
        pool.reserve(&InferenceExecutionIdentity::default(), &admission),
        Err(WorkingMemoryError::UnknownBound)
    ));
    let other = ledger(false, [MemoryLimit::Unlimited; 3]);
    assert!(prepared
        .register_storage([(1u32, allocation(&other, gpu(0), 1))])
        .is_err());
    assert_eq!(pool.snapshot().unwrap(), before);
    let unquoted = pool.acquire_unquoted().unwrap();
    let excluded = pool.snapshot().unwrap();
    assert!(matches!(
        pool.reserve(&InferenceExecutionIdentity::default(), &request(&pool, &[])),
        Err(WorkingMemoryError::UnknownBound)
    ));
    assert_eq!(pool.snapshot().unwrap(), excluded);
    drop(unquoted);
    let mut expected = before;
    expected.domains[0].historical_peak_bytes = excluded.domains[0].historical_peak_bytes;
    assert_eq!(pool.snapshot().unwrap(), expected);
}

#[test]
fn small_reference_model_tracks_reservations_shared_storage_pins_and_independent_copies() {
    use std::collections::BTreeMap;
    for unified in [false, true] {
        let pool = ledger(unified, [MemoryLimit::Unlimited; 3]);
        let before = pool.snapshot().unwrap();
        let mut owners = Vec::new();
        let mut reservations = Vec::new();
        let mut model = BTreeMap::<u32, (MemoryLocation, u64, usize)>::new();
        let mut random = 71u32;
        for _ in 0..200 {
            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
            let key = random % 13;
            let location = if key % 3 == 0 {
                MemoryLocation::Host
            } else {
                gpu(key % 2)
            };
            let bytes = u64::from(key) + 1;
            if random % 7 == 0 {
                let reservation = pool
                    .reserve(
                        &InferenceExecutionIdentity::default(),
                        &request(&pool, &[(location, bytes)]),
                    )
                    .unwrap();
                reservations.push((location, bytes, reservation));
            } else if random % 7 == 1 && !reservations.is_empty() {
                drop(reservations.swap_remove(random as usize % reservations.len()));
            } else if random % 4 == 0 && !owners.is_empty() {
                let (retired, owner) = owners.swap_remove(random as usize % owners.len());
                drop(owner);
                let count = &mut model.get_mut(&retired).unwrap().2;
                *count -= 1;
                if *count == 0 {
                    model.remove(&retired);
                }
            } else {
                let owner = if model.contains_key(&key) {
                    StoragePublicationLayout::new(1)
                        .unwrap()
                        .fund(&pool)
                        .unwrap()
                        .pin_registered_storage([(key, bytes)])
                        .unwrap()
                } else {
                    StoragePublicationLayout::new(1)
                        .unwrap()
                        .fund(&pool)
                        .unwrap()
                        .register_storage([(key, allocation(&pool, location, bytes))])
                        .unwrap()
                };
                model.entry(key).or_insert((location, bytes, 0)).2 += 1;
                owners.push((key, owner));
            }
            let snapshot = pool.snapshot().unwrap();
            for domain in &snapshot.domains {
                let retained: u64 = model
                    .values()
                    .filter(|(location, _, _)| {
                        pool.topology().domain_for(*location).unwrap() == domain.domain
                    })
                    .map(|(_, bytes, _)| *bytes)
                    .sum();
                let reserved: u64 = reservations
                    .iter()
                    .filter(|(location, _, _)| {
                        pool.topology().domain_for(*location).unwrap() == domain.domain
                    })
                    .map(|(_, bytes, _)| *bytes)
                    .sum();
                let expected = retained.checked_add(reserved).unwrap();
                let baseline = before
                    .domains
                    .iter()
                    .find(|item| item.domain == domain.domain)
                    .unwrap();
                assert_eq!(
                    domain.current_charge_bytes
                        - domain.registry_metadata_bytes
                        - domain.reservation_control_bytes
                        - baseline.current_charge_bytes,
                    expected
                );
            }
        }
        drop((owners, reservations));
        assert!(pool
            .snapshot()
            .unwrap()
            .domains
            .iter()
            .zip(&before.domains)
            .all(|(a, b)| a.current_charge_bytes == b.current_charge_bytes));
    }
}

#[test]
fn fixed_host_bookkeeping_is_charged_under_all_limit_modes() {
    let topology = topology(false);
    let baseline = DomainMemoryRequirements::zero(&topology);
    let unlimited = MemoryLimits::unlimited(&topology);
    let bytes = MemoryLedger::fixed_owner_bytes(&topology, &unlimited, &baseline).unwrap();
    assert!(bytes > 0);
    for limit in [MemoryLimit::Unlimited, MemoryLimit::Finite(bytes)] {
        let limits = MemoryLimits::resolve(&topology, [(topology.host_domain(), limit)]).unwrap();
        let pool = MemoryLedger::new(Arc::clone(&topology), limits, baseline.clone()).unwrap();
        let snapshot = pool.snapshot().unwrap();
        assert_eq!(snapshot.domains[0].fixed_baseline.accounted_bytes, bytes);
        assert_eq!(snapshot.domains[0].current_charge_bytes, bytes);
    }
    let limits = MemoryLimits::resolve(
        &topology,
        [(topology.host_domain(), MemoryLimit::Finite(bytes - 1))],
    )
    .unwrap();
    assert!(matches!(
        MemoryLedger::new(topology, limits, baseline),
        Err(WorkingMemoryError::Domain(
            MemoryDomainError::BudgetExceeded { .. }
        ))
    ));
}

#[test]
fn prospective_allowance_converts_to_fixed_backing_and_restores_its_origin_on_reuse() {
    let pool = ledger(false, [MemoryLimit::Unlimited; 3]);
    let baseline = pool.snapshot().unwrap();
    let placement = MemoryPlacement::possible_locations(
        pool.topology(),
        [MemoryLocation::Host, gpu(0), gpu(1)],
        "fixture allocator may choose fixed or managed backing".into(),
    )
    .unwrap();
    let mut requirements = DomainMemoryRequirements::zero(pool.topology());
    requirements.add_allocation(64, &placement).unwrap();
    requirements
        .add_allocation(
            MemoryLedger::storage_metadata_control_bytes().unwrap()
                + StoragePublicationLayout::<u32>::new(2)
                    .unwrap()
                    .requested_bytes(),
            &pool.host_placement_handle(),
        )
        .unwrap();
    let mut admission = request(&pool, &[]);
    admission
        .state
        .execution_workspace
        .as_mut()
        .unwrap()
        .physical_domains
        .as_mut()
        .unwrap()
        .activations = requirements;
    let (metadata, run) = pool
        .reserve(&InferenceExecutionIdentity::default(), &admission)
        .unwrap()
        .into_funding()
        .unwrap();
    let scope = run.scope().unwrap();
    let fixed = StoragePublicationLayout::new(2)
        .unwrap()
        .fund_from(&scope)
        .unwrap()
        .adopt_storage_individually(
            &scope,
            [(1u32, allocation(&pool, MemoryLocation::Host, 32))],
        )
        .unwrap();
    let snapshot = pool.snapshot().unwrap();
    assert_eq!(snapshot.domains[0].estimated_placement_allowance_bytes, 32);
    assert_eq!(snapshot.domains[1].estimated_placement_allowance_bytes, 64);
    drop(fixed);
    assert_eq!(
        pool.snapshot().unwrap().domains[0].estimated_placement_allowance_bytes,
        64
    );
    let managed = StoragePublicationLayout::new(2)
        .unwrap()
        .fund_from(&scope)
        .unwrap()
        .adopt_storage_individually(
            &scope,
            [(2u32, StorageAllocation::new(64, Arc::new(placement)))],
        )
        .unwrap();
    run.close().unwrap();
    scope.certify().unwrap();
    drop(metadata);
    for domain in pool.snapshot().unwrap().domains {
        assert_eq!(domain.estimated_placement_allowance_bytes, 64);
    }
    drop(managed);
    assert_eq!(allocation_delta(&pool, &baseline, MemoryLocation::Host), 0);
    assert!(pool
        .snapshot()
        .unwrap()
        .domains
        .iter()
        .all(|domain| domain.estimated_placement_allowance_bytes == 0));
}

#[test]
fn finite_and_unlimited_require_consistent_complete_execution_reports() {
    for unified in [false, true] {
        for limit in [MemoryLimit::Finite(1_000_000), MemoryLimit::Unlimited] {
            let pool = ledger(unified, [limit; 3]);
            let mut admission = request(&pool, &[(gpu(0), 64)]);
            let before = pool.snapshot().unwrap();
            admission.incremental_required_bytes = Some(63);
            assert!(matches!(
                pool.reserve(&InferenceExecutionIdentity::default(), &admission),
                Err(WorkingMemoryError::IdentityMismatch)
            ));
            assert_eq!(pool.snapshot().unwrap(), before);
            admission.incremental_required_bytes = Some(64);
            admission
                .state
                .execution_workspace
                .as_mut()
                .unwrap()
                .activations = WorkspaceBound::Unknown {
                reason: "selected mechanism supplied no complete allocation witness".into(),
            };
            assert!(matches!(
                pool.reserve(&InferenceExecutionIdentity::default(), &admission),
                Err(WorkingMemoryError::UnknownBound)
            ));
            assert_eq!(pool.snapshot().unwrap(), before);
        }
    }
}

#[test]
fn simultaneous_transactions_compete_for_exact_host_and_device_allowances() {
    for unified in [false, true] {
        let topology = topology(unified);
        let baseline = DomainMemoryRequirements::zero(&topology);
        let unconstrained = MemoryLimits::unlimited(&topology);
        let fixed = MemoryLedger::fixed_owner_bytes(&topology, &unconstrained, &baseline).unwrap();
        let probe = MemoryLedger::new(topology.clone(), unconstrained, baseline.clone()).unwrap();
        let admission = request(
            &probe,
            &[(MemoryLocation::Host, 16), (gpu(0), 64), (gpu(1), 64)],
        );
        let requirements = probe.reservation_requirements(&admission, None).unwrap();
        let host = topology.host_domain();
        let host_increment = requirements.get(host).unwrap().total().unwrap();
        let limits = MemoryLimits::resolve(
            &topology,
            topology.domains().map(|(domain, _)| {
                (
                    domain,
                    MemoryLimit::Finite(if domain == host {
                        fixed.checked_add(host_increment).unwrap()
                    } else {
                        requirements.get(domain).unwrap().total().unwrap()
                    }),
                )
            }),
        )
        .unwrap();
        drop(probe);
        let pool = MemoryLedger::new(topology, limits, baseline).unwrap();
        let before = pool.snapshot().unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let outcomes = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..2)
                .map(|_| {
                    let barrier = barrier.clone();
                    let pool = pool.clone();
                    let admission = &admission;
                    scope.spawn(move || {
                        let execution = InferenceExecutionIdentity::default();
                        barrier.wait();
                        let result = pool.reserve(&execution, admission);
                        // Keep the successful account alive until both attempts finish.
                        barrier.wait();
                        result
                    })
                })
                .collect();
            barrier.wait();
            barrier.wait();
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(outcomes.iter().filter(|result| result.is_ok()).count(), 1);
        let rejected = outcomes
            .iter()
            .find_map(|result| result.as_ref().err())
            .unwrap();
        assert!(
            matches!(rejected, WorkingMemoryError::Domain(MemoryDomainError::BudgetExceeded { domain, .. }) if *domain == host)
        );
        let committed = pool.snapshot().unwrap();
        for (initial, current) in before.domains.iter().zip(&committed.domains) {
            assert_eq!(
                current.current_charge_bytes,
                initial.current_charge_bytes
                    + requirements.get(current.domain).unwrap().total().unwrap()
            );
            assert_eq!(current.historical_peak_bytes, current.current_charge_bytes);
        }
        drop(outcomes);
        let retired = pool.snapshot().unwrap();
        assert_eq!(retired.funding_accounts, 0);
        for (initial, current) in before.domains.iter().zip(&retired.domains) {
            assert_eq!(current.current_charge_bytes, initial.current_charge_bytes);
        }
    }
}

#[test]
fn allocation_observation_reports_placement_without_retaining_or_admitting_storage() {
    for unified in [false, true] {
        let pool = ledger(unified, [MemoryLimit::Unlimited; 3]);
        let expected = allocation(&pool, gpu(1), 64);
        let storage = StoragePublicationLayout::new(1)
            .unwrap()
            .fund(&pool)
            .unwrap()
            .register_storage([(42u32, expected.clone())])
            .unwrap();
        let before = pool.snapshot().unwrap();
        let descriptor = pool.registered_allocation(&42u32).unwrap().unwrap();
        assert_eq!(descriptor, expected);
        assert_eq!(pool.snapshot().unwrap(), before);
        assert!(pool.registered_allocation(&42u64).unwrap().is_none());
        drop(storage);
        assert!(pool.registered_allocation(&42u32).unwrap().is_none());
        assert_eq!(descriptor.capacity_bytes(), 64);
    }
}

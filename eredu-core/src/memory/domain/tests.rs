use super::*;

fn device(ordinal: u32) -> MemoryLocation {
    MemoryLocation::Device(MemoryDeviceId {
        backend: "fixture",
        ordinal,
    })
}

fn topology(unified: bool) -> MemoryTopology {
    MemoryTopology::new(if unified {
        vec![MemoryDomainDescription {
            name: "shared".into(),
            locations: vec![MemoryLocation::Host, device(0), device(1)],
        }]
    } else {
        vec![
            MemoryDomainDescription {
                name: "host".into(),
                locations: vec![MemoryLocation::Host],
            },
            MemoryDomainDescription {
                name: "device0".into(),
                locations: vec![device(0)],
            },
            MemoryDomainDescription {
                name: "device1".into(),
                locations: vec![device(1)],
            },
        ]
    })
    .unwrap()
}

fn fixed(topology: &MemoryTopology, location: MemoryLocation) -> MemoryPlacement {
    MemoryPlacement::fixed(topology, topology.domain_for(location).unwrap()).unwrap()
}

#[test]
fn topology_is_physical_sharing_without_allocation_sharing() {
    for unified in [false, true] {
        let topology = topology(unified);
        let host = topology.host_domain();
        let accelerator = topology.domain_for(device(0)).unwrap();
        assert_eq!(host == accelerator, unified);
        let mut requirements = DomainMemoryRequirements::zero(&topology);
        requirements
            .add_allocation(64, &fixed(&topology, MemoryLocation::Host))
            .unwrap();
        requirements
            .add_allocation(16, &fixed(&topology, MemoryLocation::Host))
            .unwrap();
        requirements
            .add_allocation(64, &fixed(&topology, device(0)))
            .unwrap();
        assert_eq!(
            requirements.get(host).unwrap().accounted_bytes,
            if unified { 144 } else { 80 }
        );
        assert_eq!(
            requirements.get(accelerator).unwrap().accounted_bytes,
            if unified { 144 } else { 64 }
        );
    }
}

#[test]
fn independent_topologies_with_identical_names_are_foreign() {
    let first = topology(false);
    let second = topology(false);
    assert_eq!(
        first.slot(second.host_domain()),
        Err(MemoryDomainError::ForeignTopology)
    );
    assert_eq!(
        MemoryLimits::unlimited(&first).validate(&second),
        Err(MemoryDomainError::ForeignTopology)
    );
    assert_eq!(
        MemoryLimits::resolve(&first, [(second.host_domain(), MemoryLimit::Unlimited)]),
        Err(MemoryDomainError::ForeignTopology)
    );
    let mut requirements = DomainMemoryRequirements::zero(&first);
    let before = requirements.clone();
    assert_eq!(
        requirements.add_allocation(1, &fixed(&second, MemoryLocation::Host)),
        Err(MemoryDomainError::ForeignTopology)
    );
    assert_eq!(requirements, before);
    assert_eq!(
        requirements.checked_add(&DomainMemoryRequirements::zero(&second)),
        Err(MemoryDomainError::ForeignTopology)
    );
}

#[test]
fn configuration_resolves_every_domain_and_rejects_duplicate_or_unknown_names() {
    let topology = topology(false);
    let configured = MemoryLimits::resolve_named(
        &topology,
        [
            ("host", MemoryLimit::Finite(10)),
            ("device1", MemoryLimit::Finite(0)),
        ],
    )
    .unwrap();
    assert_eq!(
        configured.get(topology.host_domain()).unwrap(),
        MemoryLimit::Finite(10)
    );
    assert_eq!(
        configured
            .get(topology.domain_for(device(0)).unwrap())
            .unwrap(),
        MemoryLimit::Unlimited
    );
    assert_eq!(configured.iter().len(), 3);
    assert!(matches!(
        MemoryLimits::resolve_named(
            &topology,
            [
                ("host", MemoryLimit::Unlimited),
                ("host", MemoryLimit::Unlimited)
            ]
        ),
        Err(MemoryDomainError::DuplicateLimit { .. })
    ));
    assert_eq!(
        MemoryLimits::resolve_named(&topology, [("other", MemoryLimit::Unlimited)]),
        Err(MemoryDomainError::UnknownDomainName { declaration: 0 })
    );
}

#[test]
fn invalid_topologies_do_not_supply_placement() {
    let domain = |name: &str, locations: Vec<_>| MemoryDomainDescription {
        name: name.into(),
        locations,
    };
    assert!(matches!(
        MemoryTopology::new(vec![]),
        Err(MemoryDomainError::EmptyTopology)
    ));
    assert!(matches!(
        MemoryTopology::new(vec![domain("host=42", vec![MemoryLocation::Host])]),
        Err(MemoryDomainError::InvalidDomainName { .. })
    ));
    assert!(matches!(
        MemoryTopology::new(vec![
            domain("host", vec![MemoryLocation::Host]),
            domain("host", vec![device(0)])
        ]),
        Err(MemoryDomainError::DuplicateDomainName { .. })
    ));
    assert!(matches!(
        MemoryTopology::new(vec![domain("host", vec![])]),
        Err(MemoryDomainError::EmptyLocations { .. })
    ));
    assert!(matches!(
        MemoryTopology::new(vec![domain("device", vec![device(0)])]),
        Err(MemoryDomainError::MissingHost)
    ));
    assert!(matches!(
        MemoryTopology::new(vec![
            domain("host", vec![MemoryLocation::Host, device(0)]),
            domain("device", vec![device(0)])
        ]),
        Err(MemoryDomainError::DuplicateLocation { .. })
    ));
    assert!(matches!(
        MemoryTopology::new(vec![domain(
            "host",
            vec![MemoryLocation::Host, MemoryLocation::Host]
        )]),
        Err(MemoryDomainError::DuplicateLocation { .. })
    ));
    assert!(matches!(
        MemoryTopology::new(vec![domain(
            "host",
            vec![
                MemoryLocation::Host,
                MemoryLocation::Device(MemoryDeviceId {
                    backend: " ",
                    ordinal: 0
                })
            ]
        )]),
        Err(MemoryDomainError::InvalidDevice)
    ));
}

#[test]
fn finite_maximum_is_not_unlimited_and_succession_checks_raises() {
    let maximum = MemoryLimit::Finite(u64::MAX);
    assert_ne!(maximum, MemoryLimit::Unlimited);
    assert!(maximum.is_raised_by(MemoryLimit::Unlimited));
    assert!(!MemoryLimit::Unlimited.is_raised_by(maximum));
    assert!(!maximum.is_raised_by(maximum));
    assert_eq!(maximum.minimum(MemoryLimit::Unlimited), maximum);
    let domain = topology(true).host_domain();
    assert_eq!(MemoryLimit::Finite(0).check(domain, 0, 0), Ok(0));
    assert!(matches!(
        MemoryLimit::Finite(0).check(domain, 0, 1),
        Err(MemoryDomainError::BudgetExceeded {
            limit_bytes: 0,
            existing_bytes: 0,
            requested_bytes: 1,
            ..
        })
    ));
    for limit in [maximum, MemoryLimit::Unlimited] {
        assert_eq!(limit.check(domain, u64::MAX, 0), Ok(u64::MAX));
        assert_eq!(
            limit.check(domain, u64::MAX, 1),
            Err(MemoryDomainError::Overflow)
        );
    }
}

#[test]
fn same_capacity_comparison_covers_exact_fit_exhaustion_zero_and_unlimited() {
    for unified in [false, true] {
        let topology = topology(unified);
        for limit in [
            MemoryLimit::Finite(0),
            MemoryLimit::Finite(144),
            MemoryLimit::Finite(145),
            MemoryLimit::Unlimited,
        ] {
            let limits = MemoryLimits::resolve(
                &topology,
                topology.domains().map(|(domain, _)| (domain, limit)),
            )
            .unwrap();
            let mut existing = DomainMemoryRequirements::zero(&topology);
            existing
                .add_allocation(64, &fixed(&topology, MemoryLocation::Host))
                .unwrap();
            let mut increment = DomainMemoryRequirements::zero(&topology);
            increment
                .add_allocation(16, &fixed(&topology, MemoryLocation::Host))
                .unwrap();
            increment
                .add_allocation(64, &fixed(&topology, device(0)))
                .unwrap();
            let result = increment.check_increment(&existing, &limits);
            assert_eq!(result.is_ok(), limit != MemoryLimit::Finite(0));
            if unified && limit == MemoryLimit::Finite(144) {
                increment.add_headroom(topology.host_domain(), 1).unwrap();
                assert_eq!(
                    increment.check_increment(&existing, &limits),
                    Err(MemoryDomainError::BudgetExceeded {
                        domain: topology.host_domain(),
                        limit_bytes: 144,
                        existing_bytes: 64,
                        requested_bytes: 81,
                    })
                );
            }
        }
    }
}

#[test]
fn mixed_limits_keep_each_accelerator_independent() {
    let topology = topology(false);
    let first = topology.domain_for(device(0)).unwrap();
    let second = topology.domain_for(device(1)).unwrap();
    let limits = MemoryLimits::resolve(
        &topology,
        [
            (first, MemoryLimit::Finite(64)),
            (second, MemoryLimit::Finite(7)),
        ],
    )
    .unwrap();
    let existing = DomainMemoryRequirements::zero(&topology);
    let mut increment = DomainMemoryRequirements::zero(&topology);
    increment
        .add_allocation(1024, &fixed(&topology, MemoryLocation::Host))
        .unwrap();
    increment
        .add_allocation(64, &fixed(&topology, device(0)))
        .unwrap();
    increment
        .add_allocation(8, &fixed(&topology, device(1)))
        .unwrap();
    let before = increment.clone();
    assert_eq!(
        increment.check_increment(&existing, &limits),
        Err(MemoryDomainError::BudgetExceeded {
            domain: second,
            limit_bytes: 7,
            existing_bytes: 0,
            requested_bytes: 8,
        })
    );
    assert_eq!(increment, before);
    assert_eq!(existing.get(first).unwrap(), DomainMemoryCharge::default());
}

#[test]
fn separate_domains_have_no_aggregate_numeric_ceiling() {
    let topology = topology(false);
    let mut requirements = DomainMemoryRequirements::zero(&topology);
    for location in [MemoryLocation::Host, device(0), device(1)] {
        requirements
            .add_allocation(u64::MAX, &fixed(&topology, location))
            .unwrap();
    }
    let limits = MemoryLimits::resolve(
        &topology,
        topology
            .domains()
            .map(|(domain, _)| (domain, MemoryLimit::Finite(u64::MAX))),
    )
    .unwrap();
    assert_eq!(
        requirements.check_increment(&DomainMemoryRequirements::zero(&topology), &limits),
        Ok(())
    );
}

#[test]
fn managed_allowances_collapse_locations_but_keep_basis_and_independent_charges() {
    for unified in [false, true] {
        let topology = topology(unified);
        let placement = MemoryPlacement::possible_locations(
            &topology,
            [MemoryLocation::Host, device(0), device(1), device(0)],
            "fixture managed allocation can migrate among registered locations".into(),
        )
        .unwrap();
        assert_eq!(placement.domains().len(), if unified { 1 } else { 3 });
        let mut requirements = DomainMemoryRequirements::zero(&topology);
        requirements.add_allocation(64, &placement).unwrap();
        requirements.add_allocation(64, &placement).unwrap();
        for (_, charge) in requirements.iter() {
            assert_eq!(charge.accounted_bytes, 0);
            assert_eq!(charge.placement_allowance_bytes, 128);
        }
        assert_eq!(requirements.placement_allowances().len(), 2);
        assert_eq!(requirements.placement_allowances()[0].placement, placement);
        assert!(requirements.backing_bytes().unwrap() > 0);
    }
}

#[test]
fn candidate_sets_require_provenance_and_reject_unaccounted_access() {
    let topology = topology(false);
    assert_eq!(
        MemoryPlacement::possible(&topology, vec![], "known".into()),
        Err(MemoryDomainError::InvalidPlacement)
    );
    assert_eq!(
        MemoryPlacement::possible(&topology, vec![topology.host_domain()], " ".into()),
        Err(MemoryDomainError::InvalidPlacement)
    );
    let placement = MemoryPlacement::possible_locations(
        &topology,
        [MemoryLocation::Host, device(0)],
        "finite fixture candidate set".into(),
    )
    .unwrap();
    assert_eq!(placement.validate_access(&topology, device(0)), Ok(()));
    assert!(placement.validate_access(&topology, device(1)).is_err());
    assert!(matches!(
        placement.validate_access(&topology, device(2)),
        Err(MemoryDomainError::UnknownLocation { .. })
    ));
}

#[test]
fn overflow_in_final_candidate_leaves_every_category_and_basis_unchanged() {
    let topology = topology(false);
    let mut requirements = DomainMemoryRequirements::zero(&topology);
    let last = topology.domain_for(device(1)).unwrap();
    requirements.add_headroom(last, u64::MAX).unwrap();
    let before = requirements.clone();
    let managed = MemoryPlacement::possible_locations(
        &topology,
        [MemoryLocation::Host, device(0), device(1)],
        "fixture candidates".into(),
    )
    .unwrap();
    assert_eq!(
        requirements.add_allocation(1, &managed),
        Err(MemoryDomainError::Overflow)
    );
    assert_eq!(requirements, before);
    let empty = DomainMemoryRequirements::zero(&topology);
    assert_eq!(
        requirements.check_increment(&empty, &MemoryLimits::unlimited(&topology)),
        Ok(())
    );
    assert_eq!(
        requirements.check_increment(&requirements, &MemoryLimits::unlimited(&topology)),
        Err(MemoryDomainError::Overflow)
    );
}

#[test]
fn category_arithmetic_cannot_hide_overflow_or_underflow() {
    let first = DomainMemoryCharge {
        accounted_bytes: u64::MAX,
        ..Default::default()
    };
    let second = DomainMemoryCharge {
        headroom_bytes: 1,
        ..Default::default()
    };
    assert_eq!(first.checked_add(second), Err(MemoryDomainError::Overflow));
    assert_eq!(first.checked_sub(second), Err(MemoryDomainError::Overflow));
    assert_eq!(first.checked_sub(first), Ok(DomainMemoryCharge::default()));
}

#[test]
fn live_limit_intersections_are_pointwise() {
    let topology = topology(false);
    let host = topology.host_domain();
    let gpu = topology.domain_for(device(0)).unwrap();
    let first = MemoryLimits::resolve(&topology, [(host, MemoryLimit::Finite(100))]).unwrap();
    let second = MemoryLimits::resolve(
        &topology,
        [
            (host, MemoryLimit::Finite(200)),
            (gpu, MemoryLimit::Finite(10)),
        ],
    )
    .unwrap();
    let effective = first.intersection(&second).unwrap();
    assert_eq!(effective.get(host).unwrap(), MemoryLimit::Finite(100));
    assert_eq!(effective.get(gpu).unwrap(), MemoryLimit::Finite(10));
    assert_eq!(
        effective
            .get(topology.domain_for(device(1)).unwrap())
            .unwrap(),
        MemoryLimit::Unlimited
    );
}

#[test]
fn finite_estimates_keep_ranges_basis_and_disjoint_headroom() {
    let topology = topology(false);
    let mut requirements = DomainMemoryRequirements::zero(&topology);
    let estimate = DomainOverheadEstimate {
        domain: topology.host_domain(),
        source: "fixture parser".into(),
        range: crate::FiniteMemoryEstimate::new(2, 9).unwrap(),
        basis: "input-dependent fixture estimate".into(),
    };
    requirements.add_estimate(estimate.clone()).unwrap();
    requirements.add_estimate(estimate.clone()).unwrap();
    requirements
        .add_headroom(topology.host_domain(), 3)
        .unwrap();
    let charge = requirements.get(topology.host_domain()).unwrap();
    assert_eq!(charge.estimated_overhead_bytes, 18);
    assert_eq!(charge.headroom_bytes, 3);
    assert_eq!(charge.total().unwrap(), 21);
    assert_eq!(
        requirements.overhead_estimates(),
        &[estimate.clone(), estimate]
    );
    let before = requirements.clone();
    assert_eq!(
        requirements.add_estimate(DomainOverheadEstimate {
            domain: topology.host_domain(),
            source: "".into(),
            range: crate::FiniteMemoryEstimate::new(0, 0).unwrap(),
            basis: "known".into(),
        }),
        Err(MemoryDomainError::InvalidEstimateDescription)
    );
    assert_eq!(requirements, before);
}

#[test]
fn checked_accumulation_matches_wide_integer_reference() {
    for unified in [false, true] {
        let topology = topology(unified);
        let mut requirements = DomainMemoryRequirements::zero(&topology);
        let mut reference = vec![[0u128; 4]; topology.len()];
        let mut random = 0x8f63_1c7d_56a0_e291u64;
        for iteration in 0..300 {
            // Reproducible PRNG arithmetic, independent of accounting arithmetic.
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            let location = [MemoryLocation::Host, device(0), device(1)][iteration % 3];
            let domain = topology.domain_for(location).unwrap();
            let bytes = if iteration % 17 == 0 {
                u64::MAX - random % 1024
            } else {
                random % 1024
            };
            let managed = iteration % 5 == 0;
            let headroom = iteration % 7 == 0 && !managed;
            let placement = if managed {
                MemoryPlacement::possible_locations(
                    &topology,
                    [MemoryLocation::Host, device(0), device(1)],
                    "fixture candidate set".into(),
                )
                .unwrap()
            } else {
                fixed(&topology, location)
            };
            let mut next = reference.clone();
            if headroom {
                next[topology.slot(domain).unwrap()][3] += u128::from(bytes);
            } else {
                for &domain in placement.domains() {
                    next[topology.slot(domain).unwrap()][usize::from(managed)] += u128::from(bytes);
                }
            }
            let fits = next
                .iter()
                .all(|row| row.iter().sum::<u128>() <= u128::from(u64::MAX));
            let before = requirements.clone();
            let result = if headroom {
                requirements.add_headroom(domain, bytes)
            } else {
                requirements.add_allocation(bytes, &placement)
            };
            assert_eq!(result.is_ok(), fits);
            if fits {
                reference = next;
            } else {
                assert_eq!(requirements, before);
            }
            for (domain, charge) in requirements.iter() {
                let actual = [
                    charge.accounted_bytes,
                    charge.placement_allowance_bytes,
                    charge.estimated_overhead_bytes,
                    charge.headroom_bytes,
                ]
                .map(u128::from);
                assert_eq!(actual, reference[topology.slot(domain).unwrap()]);
            }
        }
    }
}

#[test]
fn resolving_into_funded_limits_preserves_all_domains_on_rejection() {
    let topology = topology(false);
    let mut target =
        MemoryLimits::resolve_named(&topology, [("host", MemoryLimit::Finite(17))]).unwrap();
    let before = target.clone();
    for declarations in [
        MemoryLimitDeclarations::new([
            ("host".into(), MemoryLimit::Finite(9)),
            ("missing".into(), MemoryLimit::Unlimited),
        ]),
        MemoryLimitDeclarations::new([
            ("host".into(), MemoryLimit::Finite(9)),
            ("host".into(), MemoryLimit::Unlimited),
        ]),
    ] {
        assert!(target
            .resolve_named_in_place(&topology, &declarations, None)
            .is_err());
        assert_eq!(target, before);
    }
    let foreign = MemoryTopology::new(vec![MemoryDomainDescription {
        name: "host".into(),
        locations: vec![MemoryLocation::Host],
    }])
    .unwrap();
    assert_eq!(
        target.resolve_named_in_place(
            &topology,
            &MemoryLimitDeclarations::default(),
            Some(&MemoryLimits::unlimited(&foreign))
        ),
        Err(MemoryDomainError::ForeignTopology)
    );
    assert_eq!(target, before);
    let ceiling =
        MemoryLimits::resolve_named(&topology, [("device1", MemoryLimit::Finite(12))]).unwrap();
    target
        .resolve_named_in_place(
            &topology,
            &MemoryLimitDeclarations::new([("host".into(), MemoryLimit::Finite(u64::MAX))]),
            Some(&ceiling),
        )
        .unwrap();
    assert_eq!(
        target.get(topology.host_domain()).unwrap(),
        MemoryLimit::Finite(u64::MAX)
    );
    assert_eq!(
        target.get(topology.domain_for(device(0)).unwrap()).unwrap(),
        MemoryLimit::Unlimited
    );
    assert_eq!(
        target.get(topology.domain_for(device(1)).unwrap()).unwrap(),
        MemoryLimit::Finite(12)
    );
}

#[test]
fn multi_producer_sum_preserves_placement_and_checks_the_final_domain() {
    for unified in [false, true] {
        let topology = topology(unified);
        let mut source = DomainMemoryRequirements::zero(&topology);
        source
            .add_allocation(64, &fixed(&topology, MemoryLocation::Host))
            .unwrap();
        let placement = MemoryPlacement::possible_locations(
            &topology,
            [MemoryLocation::Host, device(1)],
            "fixture managed backing".into(),
        )
        .unwrap();
        let mut destination = DomainMemoryRequirements::zero(&topology);
        destination.add_allocation(32, &placement).unwrap();
        let combined =
            DomainMemoryRequirements::checked_sum(&topology, &[&source, &destination]).unwrap();
        assert_eq!(combined, source.checked_add(&destination).unwrap());
        assert_eq!(
            combined.clone_backing_bytes().unwrap(),
            combined.clone().backing_bytes().unwrap()
        );
        let mut maximal = DomainMemoryRequirements::zero(&topology);
        maximal
            .add_allocation(u64::MAX, &fixed(&topology, device(1)))
            .unwrap();
        assert_eq!(
            DomainMemoryRequirements::checked_sum(&topology, &[&source, &destination, &maximal]),
            Err(MemoryDomainError::Overflow)
        );
        assert_eq!(
            source.get(topology.host_domain()).unwrap().accounted_bytes,
            64
        );
        assert_eq!(
            destination
                .get(topology.domain_for(device(1)).unwrap())
                .unwrap()
                .placement_allowance_bytes,
            32
        );
    }
}

#[test]
fn descriptive_component_removal_preserves_independent_equal_producers() {
    for unified in [false, true] {
        let topology = topology(unified);
        let placement = MemoryPlacement::possible_locations(
            &topology,
            [MemoryLocation::Host, device(0)],
            "backend candidate set".into(),
        )
        .unwrap();
        let mut component = DomainMemoryRequirements::zero(&topology);
        component.add_allocation(64, &placement).unwrap();
        component
            .add_allocation(11, &fixed(&topology, MemoryLocation::Host))
            .unwrap();
        component.add_headroom(topology.host_domain(), 3).unwrap();
        component
            .add_estimate(DomainOverheadEstimate {
                domain: topology.host_domain(),
                source: "producer".into(),
                range: crate::FiniteMemoryEstimate::new(2, 7).unwrap(),
                basis: "finite native controls".into(),
            })
            .unwrap();
        let mut combined = component.checked_add(&component).unwrap();
        combined.subtract_component(&component).unwrap();
        assert_eq!(combined, component);
        combined.subtract_component(&component).unwrap();
        assert!(combined
            .iter()
            .all(|(_, charge)| charge.total().unwrap() == 0));
        assert!(combined.placement_allowances().is_empty());
        assert!(combined.overhead_estimates().is_empty());
    }
}

#[test]
fn descriptive_component_final_domain_underflow_is_atomic() {
    let topology = topology(false);
    let mut whole = DomainMemoryRequirements::zero(&topology);
    whole
        .add_allocation(10, &fixed(&topology, MemoryLocation::Host))
        .unwrap();
    whole
        .add_allocation(4, &fixed(&topology, device(1)))
        .unwrap();
    let mut part = DomainMemoryRequirements::zero(&topology);
    part.add_allocation(10, &fixed(&topology, MemoryLocation::Host))
        .unwrap();
    part.add_allocation(5, &fixed(&topology, device(1)))
        .unwrap();
    let before = whole.clone();
    assert_eq!(
        whole.subtract_component(&part),
        Err(MemoryDomainError::Overflow)
    );
    assert_eq!(whole, before);
    let foreign = DomainMemoryRequirements::zero(&self::topology(false));
    assert_eq!(
        whole.subtract_component(&foreign),
        Err(MemoryDomainError::ForeignTopology)
    );
    assert_eq!(whole, before);
}

#[test]
fn descriptive_component_requires_complete_provenance_multiplicity() {
    let topology = topology(false);
    let first = MemoryPlacement::possible_locations(
        &topology,
        [MemoryLocation::Host, device(0)],
        "first allocator".into(),
    )
    .unwrap();
    let second = MemoryPlacement::possible_locations(
        &topology,
        [MemoryLocation::Host, device(0)],
        "second allocator".into(),
    )
    .unwrap();
    let mut whole = DomainMemoryRequirements::zero(&topology);
    whole.add_allocation(64, &first).unwrap();
    whole.add_allocation(64, &second).unwrap();
    let mut part = DomainMemoryRequirements::zero(&topology);
    part.add_allocation(64, &first).unwrap();
    part.add_allocation(64, &first).unwrap();
    let before = whole.clone();
    assert_eq!(
        whole.subtract_component(&part),
        Err(MemoryDomainError::MissingRequirementComponent)
    );
    assert_eq!(whole, before);
}

#[test]
fn descriptive_component_rejects_mismatched_overhead_basis_without_mutation() {
    let topology = topology(true);
    let estimate = |basis: &str| DomainOverheadEstimate {
        domain: topology.host_domain(),
        source: "same diagnostic label".into(),
        range: crate::FiniteMemoryEstimate::new(0, 9).unwrap(),
        basis: basis.into(),
    };
    let mut whole = DomainMemoryRequirements::zero(&topology);
    whole.add_estimate(estimate("actual producer")).unwrap();
    let mut part = DomainMemoryRequirements::zero(&topology);
    part.add_estimate(estimate("different producer")).unwrap();
    let before = whole.clone();
    assert_eq!(
        whole.subtract_component(&part),
        Err(MemoryDomainError::MissingRequirementComponent)
    );
    assert_eq!(whole, before);
}

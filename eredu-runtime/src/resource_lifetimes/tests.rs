use super::*;
use crate::mechanism_resources::{describe_mechanism_resources, MechanismResourceQuery};
use eredu_nn::mechanism_memory::*;

fn id(key: &str) -> ResourceIdentity {
    ResourceIdentity {
        scope: "instance".into(),
        key: key.into(),
    }
}
fn extent(payload: u64, capacity: u64) -> ResourceExtent {
    ResourceExtent {
        payload: ResourceByteBounds::exact(payload),
        capacity: ResourceByteBounds::exact(capacity),
    }
}
fn allocation(name: &str, bytes: u64, pool: &str) -> ResourceAllocation {
    ResourceAllocation {
        identity: id(name),
        uses: vec![ResourceUse {
            owner: id("module"),
            role: ResourceRole::Workspace,
        }],
        placement: Observed::Available {
            value: id(pool),
            kind: ObservationKind::Exact,
            source: "test device".into(),
        },
        size: ResourceSize::Fixed {
            extent: extent(bytes, bytes),
        },
    }
}
fn acquired(
    allocations: Vec<ResourceAllocation>,
    hold: ResourceLifetime,
) -> ResourceLifetimeDescription {
    ResourceLifetimeDescription {
        lifetimes: allocations
            .iter()
            .map(|a| (a.identity.clone(), vec![hold.clone()]))
            .collect(),
        resources: ResourceDescription {
            schema_version: RESOURCE_DESCRIPTION_SCHEMA_VERSION,
            scope: id("description"),
            context: ResourceContext::default(),
            horizon: ResourceContext::default(),
            coverage: ResourceCoverage::Complete,
            allocations,
        },
        live_at_acquire: true,
    }
}
fn acquire(name: &str, bytes: u64, token: &str) -> ResourceLifetimeEvent {
    ResourceLifetimeEvent::Acquire(acquired(
        vec![allocation(name, bytes, "unified")],
        ResourceLifetime::NativeCompletion(id(token)),
    ))
}
fn report(events: Vec<ResourceLifetimeEvent>) -> ResourcePeakReport {
    compose_resource_peaks(&ResourceLifetimePlan {
        coverage: ResourceCoverage::Complete,
        events,
    })
    .unwrap()
}
fn peak(report: &ResourcePeakReport, pool: &str) -> (u64, Option<u64>) {
    let bound = &report
        .pools
        .iter()
        .find(|p| p.pool == id(pool))
        .unwrap()
        .peak
        .capacity;
    (bound.lower_bytes, bound.upper_bytes)
}
fn complete(token: &str) -> ResourceLifetimeEvent {
    ResourceLifetimeEvent::Complete(id(token))
}

#[test]
fn sequential_and_concurrent_work_have_different_peaks() {
    assert_eq!(
        peak(
            &report(vec![
                acquire("a", 10, "a"),
                complete("a"),
                acquire("b", 20, "b"),
                complete("b")
            ]),
            "unified"
        ),
        (20, Some(20))
    );
    assert_eq!(
        peak(
            &report(vec![
                acquire("a", 10, "a"),
                acquire("b", 20, "b"),
                complete("a"),
                complete("b")
            ]),
            "unified"
        ),
        (30, Some(30))
    );
}

#[test]
fn lazy_graphs_survive_native_completion_until_evaluation() {
    let a = acquired(
        vec![allocation("lazy", 10, "unified")],
        ResourceLifetime::Evaluation(id("graph")),
    );
    // A completion with the same label must not release the lazy dependency.
    let b = acquired(
        vec![allocation("native", 5, "unified")],
        ResourceLifetime::NativeCompletion(id("graph")),
    );
    let r = report(vec![
        ResourceLifetimeEvent::Acquire(a),
        ResourceLifetimeEvent::Acquire(b),
        complete("graph"),
        acquire("next", 20, "next"),
        complete("next"),
        ResourceLifetimeEvent::Evaluate(id("graph")),
        acquire("last", 25, "last"),
    ]);
    assert_eq!(peak(&r, "unified"), (30, Some(30)));
}

#[test]
fn explicit_evaluation_batches_bound_live_graphs() {
    let lazy = |name: &str, boundary: &str| {
        ResourceLifetimeEvent::Acquire(acquired(
            vec![allocation(name, 10, "unified")],
            ResourceLifetime::Evaluation(id(boundary)),
        ))
    };
    assert_eq!(
        peak(
            &report(vec![
                lazy("a", "first"),
                ResourceLifetimeEvent::Evaluate(id("first")),
                lazy("b", "second")
            ]),
            "unified"
        ),
        (10, Some(10))
    );
    assert_eq!(
        peak(
            &report(vec![
                lazy("a", "all"),
                lazy("b", "all"),
                ResourceLifetimeEvent::Evaluate(id("all"))
            ]),
            "unified"
        ),
        (20, Some(20))
    );
}

#[test]
fn shared_backing_lives_until_all_parameter_and_lazy_references_end() {
    let shared = allocation("converted", 10, "unified");
    let owner = acquired(
        vec![shared.clone()],
        ResourceLifetime::Owner(id("parameters")),
    );
    let lazy = acquired(vec![shared], ResourceLifetime::Evaluation(id("graph")));
    let r = report(vec![
        ResourceLifetimeEvent::Acquire(owner),
        ResourceLifetimeEvent::Acquire(lazy),
        ResourceLifetimeEvent::Release(id("parameters")),
        acquire("temporary", 20, "native"),
        complete("native"),
        ResourceLifetimeEvent::Evaluate(id("graph")),
        acquire("after", 25, "after"),
    ]);
    assert_eq!(peak(&r, "unified"), (30, Some(30)));
    assert!(r.missing.is_empty());
}

#[test]
fn disjoint_pools_and_payload_capacity_are_composed_independently() {
    let mut device = allocation("device", 10, "device");
    device.size = ResourceSize::Fixed {
        extent: extent(10, 16),
    };
    let r = report(vec![ResourceLifetimeEvent::Acquire(acquired(
        vec![device, allocation("host", 40, "host")],
        ResourceLifetime::Owner(id("owner")),
    ))]);
    assert_eq!(peak(&r, "device"), (16, Some(16)));
    assert_eq!(peak(&r, "host"), (40, Some(40)));
    assert_eq!(
        r.pools
            .iter()
            .find(|p| p.pool == id("device"))
            .unwrap()
            .peak
            .payload
            .upper_bytes,
        Some(10)
    );
}

#[test]
fn unrelated_horizon_maxima_do_not_imply_simultaneous_lower_bound() {
    let mut a = allocation("a", 0, "unified");
    a.size = ResourceSize::ContextDependent {
        current: extent(2, 2),
        horizon_peak: extent(40, 40),
    };
    let mut b = allocation("b", 0, "unified");
    b.size = ResourceSize::ContextDependent {
        current: extent(3, 3),
        horizon_peak: extent(50, 50),
    };
    let r = report(vec![ResourceLifetimeEvent::Acquire(acquired(
        vec![a, b],
        ResourceLifetime::Owner(id("state")),
    ))]);
    assert_eq!(peak(&r, "unified"), (50, Some(90)));
}

#[test]
fn current_dynamic_size_is_not_a_minimum_after_acquisition() {
    let mut a = allocation("dynamic", 0, "unified");
    a.size = ResourceSize::ContextDependent {
        current: extent(10, 10),
        horizon_peak: extent(10, 10),
    };
    let together = report(vec![ResourceLifetimeEvent::Acquire(acquired(
        vec![a.clone(), allocation("b", 20, "unified")],
        ResourceLifetime::Owner(id("owner")),
    ))]);
    let separate = report(vec![
        ResourceLifetimeEvent::Acquire(acquired(vec![a], ResourceLifetime::Owner(id("owner")))),
        acquire("b", 20, "b"),
    ]);
    assert_eq!(peak(&together, "unified"), (30, Some(30)));
    assert_eq!(peak(&separate, "unified"), (20, Some(30)));
}

#[test]
fn unknown_retention_keeps_possible_bytes_without_inventing_minimum_liveness() {
    let r = report(vec![
        ResourceLifetimeEvent::Acquire(acquired(
            vec![allocation("a", 10, "unified")],
            ResourceLifetime::Unknown,
        )),
        acquire("b", 20, "b"),
    ]);
    assert_eq!(peak(&r, "unified"), (20, None));
    assert!(r.missing.iter().any(|s| s.contains("retention")));
}

#[test]
fn unspecified_missing_and_unplaced_resources_prevent_finite_upper_bounds() {
    let unspecified = compose_resource_peaks(&ResourceLifetimePlan::default()).unwrap();
    assert!(!unspecified.missing.is_empty());
    let mut partial = acquired(
        vec![allocation("a", 10, "unified")],
        ResourceLifetime::Owner(id("owner")),
    );
    partial.resources.coverage = ResourceCoverage::Partial {
        reasons: vec!["opaque workspace".into()],
    };
    let r = report(vec![ResourceLifetimeEvent::Acquire(partial)]);
    assert_eq!(peak(&r, "unified"), (10, None));
    let mut unplaced = allocation("unplaced", 5, "ignored");
    unplaced.placement = Observed::Unavailable {
        reason: "unknown placement".into(),
    };
    let r = report(vec![
        acquire("known", 10, "known"),
        ResourceLifetimeEvent::Acquire(acquired(
            vec![unplaced],
            ResourceLifetime::Owner(id("owner")),
        )),
    ]);
    assert_eq!(peak(&r, "unified"), (10, None));
    assert_eq!(r.unplaced, vec![id("unplaced")]);
    assert_eq!(r.pools.len(), 1);
}

#[test]
fn unknown_capacity_preserves_known_payload_and_other_physical_pools() {
    let mut unknown = allocation("unknown", 10, "device");
    unknown.size = ResourceSize::Fixed {
        extent: ResourceExtent {
            payload: ResourceByteBounds::exact(10),
            capacity: ResourceByteBounds::unknown(10, "alignment unavailable"),
        },
    };
    let r = report(vec![ResourceLifetimeEvent::Acquire(acquired(
        vec![unknown, allocation("host", 20, "host")],
        ResourceLifetime::Owner(id("owner")),
    ))]);
    assert_eq!(peak(&r, "device"), (10, None));
    assert_eq!(peak(&r, "host"), (20, Some(20)));
    assert_eq!(
        r.pools
            .iter()
            .find(|p| p.pool == id("device"))
            .unwrap()
            .peak
            .payload
            .upper_bytes,
        Some(10)
    );
}

#[test]
fn incompatible_aliases_invalid_schedules_and_overflow_are_errors() {
    let bad = |events| {
        compose_resource_peaks(&ResourceLifetimePlan {
            coverage: ResourceCoverage::Complete,
            events,
        })
        .is_err()
    };
    assert!(bad(vec![acquire("a", 10, "a"), acquire("a", 20, "b")]));
    assert!(bad(vec![
        acquire("a", 10, "a"),
        ResourceLifetimeEvent::Acquire(acquired(
            vec![allocation("a", 10, "other")],
            ResourceLifetime::Owner(id("owner"))
        ))
    ]));
    assert!(bad(vec![complete("absent")]));
    assert!(bad(vec![
        acquire("a", 10, "a"),
        complete("a"),
        complete("a")
    ]));
    assert!(bad(vec![
        acquire("a", 10, "a"),
        complete("a"),
        acquire("b", 10, "a")
    ]));
    assert!(bad(vec![acquire("a", u64::MAX, "a"), acquire("b", 1, "b")]));
    let mut invalid = acquired(
        vec![allocation("a", 10, "unified")],
        ResourceLifetime::Owner(id("owner")),
    );
    invalid
        .lifetimes
        .insert(id("absent"), vec![ResourceLifetime::Unknown]);
    assert!(bad(vec![ResourceLifetimeEvent::Acquire(invalid)]));
    // Retention classes have separate namespaces even with equal labels.
    assert!(bad(vec![
        ResourceLifetimeEvent::Acquire(acquired(
            vec![allocation("a", 10, "unified")],
            ResourceLifetime::Evaluation(id("same"))
        )),
        complete("same")
    ]));
}

#[test]
fn shared_dynamic_backings_require_the_same_context_and_horizon() {
    let mut a = allocation("a", 0, "unified");
    a.size = ResourceSize::ContextDependent {
        current: extent(10, 10),
        horizon_peak: extent(20, 20),
    };
    let first = acquired(vec![a.clone()], ResourceLifetime::Owner(id("owner")));
    let mut second = acquired(vec![a], ResourceLifetime::Evaluation(id("graph")));
    second
        .resources
        .horizon
        .dimensions
        .insert("positions".into(), 10);
    assert!(compose_resource_peaks(&ResourceLifetimePlan {
        coverage: ResourceCoverage::Complete,
        events: vec![
            ResourceLifetimeEvent::Acquire(first),
            ResourceLifetimeEvent::Acquire(second)
        ]
    })
    .is_err());
}

fn storage(
    name: &str,
    bytes: u64,
    retention: StorageRetention,
    backing: MechanismBacking,
) -> MechanismStorage {
    MechanismStorage {
        name: name.into(),
        role: MechanismStorageRole::Scratch,
        payload: MechanismBytes::exact(bytes),
        capacity: MechanismBytes::exact(bytes),
        backing,
        placement: MechanismPlacement::Execution,
        retention,
        detail: "fixture allocation".into(),
    }
}
fn mechanism(storages: Vec<MechanismStorage>) -> MechanismResourceDescription {
    describe_mechanism_resources(
        MechanismMemoryContract {
            values: vec![],
            storage: storages,
            missing: vec![],
        },
        &MechanismResourceQuery {
            invocation: id("mechanism"),
            execution_pool: Observed::Available {
                value: id("unified"),
                kind: ObservationKind::Exact,
                source: "fixture".into(),
            },
            host_pool: Observed::Unavailable {
                reason: "unused".into(),
            },
            owner_backings: BTreeMap::from([("shared".into(), id("shared"))]),
        },
    )
    .unwrap()
}
fn bindings() -> MechanismLifetimeBindings {
    MechanismLifetimeBindings {
        completion: id("native"),
        evaluation: Some(id("evaluation")),
        owners: BTreeMap::from([(id("shared"), id("parameters"))]),
    }
}

#[test]
fn mechanism_bridge_preserves_alias_retention_and_internal_envelopes() {
    let shared = MechanismBacking::Owner("shared".into());
    let m = mechanism(vec![
        storage(
            "owner",
            10,
            StorageRetention::ParameterOwner,
            shared.clone(),
        ),
        storage("lazy", 10, StorageRetention::Evaluation, shared),
        storage(
            "scratch",
            20,
            StorageRetention::NativeCompletion,
            MechanismBacking::Invocation,
        ),
    ]);
    let description = describe_mechanism_lifetimes(&m, &bindings()).unwrap();
    assert!(!description.live_at_acquire);
    assert_eq!(description.lifetimes[&id("shared")].len(), 2);
    let r = report(vec![
        ResourceLifetimeEvent::Acquire(description),
        complete("native"),
        ResourceLifetimeEvent::Release(id("parameters")),
        acquire("next", 30, "next"),
        ResourceLifetimeEvent::Evaluate(id("evaluation")),
    ]);
    // Internal intermediates might not coexist; later lazy reference still adds
    // to the conservative upper, without an asserted simultaneous lower.
    assert_eq!(peak(&r, "unified"), (30, Some(40)));
}

#[test]
fn mechanism_bridge_cannot_hide_missing_aliases_or_forged_bounds() {
    let mut m = mechanism(vec![
        storage(
            "a",
            10,
            StorageRetention::NativeCompletion,
            MechanismBacking::Owner("shared".into()),
        ),
        storage(
            "lazy",
            10,
            StorageRetention::Evaluation,
            MechanismBacking::Owner("shared".into()),
        ),
    ]);
    m.storage_bindings.remove("lazy");
    let r = report(vec![
        ResourceLifetimeEvent::Acquire(describe_mechanism_lifetimes(&m, &bindings()).unwrap()),
        complete("native"),
    ]);
    assert_eq!(peak(&r, "unified"), (10, None));
    let mut m = mechanism(vec![storage(
        "a",
        10,
        StorageRetention::NativeCompletion,
        MechanismBacking::Invocation,
    )]);
    m.storage_bindings.insert(
        "made-up".into(),
        m.resources.allocations[0].identity.clone(),
    );
    assert!(describe_mechanism_lifetimes(&m, &bindings()).is_err());
    m.storage_bindings.remove("made-up");
    m.resources.allocations[0].size = ResourceSize::Fixed {
        extent: extent(1, 1),
    };
    assert!(describe_mechanism_lifetimes(&m, &bindings()).is_err());
}

#[test]
fn empty_complete_work_has_no_missing_resources() {
    assert_eq!(
        report(vec![]),
        ResourcePeakReport {
            pools: vec![],
            missing: vec![],
            unplaced: vec![]
        }
    );
    let mut description = acquired(
        vec![allocation("a", 10, "unified")],
        ResourceLifetime::Owner(id("owner")),
    );
    description.lifetimes.clear();
    let r = report(vec![ResourceLifetimeEvent::Acquire(description)]);
    assert_eq!(peak(&r, "unified"), (10, None));
}

#[test]
fn mechanism_wrapper_cannot_invent_physical_aliases_or_resolve_unknown_facts() {
    let mut m = mechanism(vec![
        storage(
            "first",
            10,
            StorageRetention::NativeCompletion,
            MechanismBacking::Invocation,
        ),
        storage(
            "second",
            10,
            StorageRetention::NativeCompletion,
            MechanismBacking::Invocation,
        ),
    ]);
    let first = m.storage_bindings["first"].clone();
    m.storage_bindings.insert("second".into(), first.clone());
    m.resources.allocations.retain(|a| a.identity == first);
    assert!(describe_mechanism_lifetimes(&m, &bindings()).is_err());
    let mut m = mechanism(vec![storage(
        "known",
        10,
        StorageRetention::NativeCompletion,
        MechanismBacking::Invocation,
    )]);
    m.contract.storage[0].backing = MechanismBacking::Unknown;
    assert!(describe_mechanism_lifetimes(&m, &bindings()).is_err());
    m.contract.storage[0].backing = MechanismBacking::Invocation;
    m.contract.storage[0].placement = MechanismPlacement::Unknown;
    assert!(describe_mechanism_lifetimes(&m, &bindings()).is_err());
}

#[test]
fn mechanism_wrapper_cannot_split_one_execution_pool() {
    let mut m = mechanism(vec![
        storage(
            "first",
            10,
            StorageRetention::NativeCompletion,
            MechanismBacking::Invocation,
        ),
        storage(
            "second",
            10,
            StorageRetention::NativeCompletion,
            MechanismBacking::Invocation,
        ),
    ]);
    m.resources.allocations[1].placement = Observed::Available {
        value: id("invented-second-pool"),
        kind: ObservationKind::Exact,
        source: "forged".into(),
    };
    assert!(describe_mechanism_lifetimes(&m, &bindings()).is_err());
}

#[test]
fn coincident_numerical_bounds_preserve_source_quality_including_later_aliases() {
    for kind in [
        ObservationKind::Estimated,
        ObservationKind::Observational,
        ObservationKind::Conservative,
    ] {
        let exact = allocation("shared", 10, "unified");
        let mut estimated = exact.clone();
        let ResourceSize::Fixed { extent } = &mut estimated.size else {
            unreachable!()
        };
        extent.payload.kind = kind;
        let first = acquired(vec![exact], ResourceLifetime::Owner(id("first")));
        let second = acquired(vec![estimated], ResourceLifetime::Owner(id("second")));
        let r = report(vec![
            ResourceLifetimeEvent::Acquire(first),
            ResourceLifetimeEvent::Acquire(second),
        ]);
        assert_eq!(r.pools[0].peak.payload.lower_bytes, 10);
        assert_eq!(r.pools[0].peak.payload.upper_bytes, Some(10));
        assert_eq!(r.pools[0].peak.payload.kind, ObservationKind::Estimated);
        assert_eq!(r.pools[0].peak.capacity.kind, ObservationKind::Exact);
    }
    let mut dynamic = allocation("dynamic", 10, "unified");
    let mut current = extent(10, 10);
    current.capacity.kind = ObservationKind::Observational;
    dynamic.size = ResourceSize::ContextDependent {
        current,
        horizon_peak: extent(10, 10),
    };
    let r = report(vec![ResourceLifetimeEvent::Acquire(acquired(
        vec![dynamic],
        ResourceLifetime::Owner(id("owner")),
    ))]);
    assert_eq!(r.pools[0].peak.payload.kind, ObservationKind::Exact);
    assert_eq!(r.pools[0].peak.capacity.kind, ObservationKind::Estimated);
}

#[test]
fn capacity_lower_includes_payload_without_adding_the_two() {
    let mut a = allocation("a", 10, "unified");
    a.size = ResourceSize::Fixed {
        extent: ResourceExtent {
            payload: ResourceByteBounds::exact(10),
            capacity: ResourceByteBounds::unknown(0, "capacity unavailable"),
        },
    };
    let r = report(vec![ResourceLifetimeEvent::Acquire(acquired(
        vec![a, allocation("b", 20, "unified")],
        ResourceLifetime::Owner(id("owner")),
    ))]);
    assert_eq!(peak(&r, "unified"), (30, None));
    assert_eq!(r.pools[0].peak.payload.upper_bytes, Some(30));
    let mut dynamic = allocation("dynamic", 0, "unified");
    dynamic.size = ResourceSize::ContextDependent {
        current: ResourceExtent {
            payload: ResourceByteBounds::exact(5),
            capacity: ResourceByteBounds::unknown(0, "start capacity"),
        },
        horizon_peak: ResourceExtent {
            payload: ResourceByteBounds::exact(50),
            capacity: ResourceByteBounds::unknown(0, "horizon capacity"),
        },
    };
    let r = report(vec![ResourceLifetimeEvent::Acquire(acquired(
        vec![dynamic],
        ResourceLifetime::Owner(id("owner")),
    ))]);
    assert_eq!(peak(&r, "unified"), (50, None));
}

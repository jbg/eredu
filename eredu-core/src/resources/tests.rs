use super::*;

fn id(scope: &str, key: &str) -> ResourceIdentity {
    ResourceIdentity {
        scope: scope.into(),
        key: key.into(),
    }
}

fn extent(payload: u64, capacity: u64) -> ResourceExtent {
    ResourceExtent {
        payload: ResourceByteBounds::exact(payload),
        capacity: ResourceByteBounds::exact(capacity),
    }
}

fn allocation(key: &str, owner: &str, role: ResourceRole) -> ResourceAllocation {
    ResourceAllocation {
        identity: id("load-1", key),
        uses: vec![ResourceUse {
            owner: id("load-1", owner),
            role,
        }],
        placement: Observed::exact(id("host-1", "shared-dram"), "native pool registry"),
        size: ResourceSize::Fixed {
            extent: extent(128, 256),
        },
    }
}

fn description() -> ResourceDescription {
    ResourceDescription {
        schema_version: RESOURCE_DESCRIPTION_SCHEMA_VERSION,
        scope: id("load-1", "execution"),
        context: ResourceContext::default(),
        horizon: ResourceContext::default(),
        coverage: ResourceCoverage::Complete,
        allocations: vec![allocation(
            "embedding-storage",
            "embedding",
            ResourceRole::Parameters,
        )],
    }
}

#[test]
fn aliases_share_backing_but_replicas_conversions_and_independent_loads_do_not() {
    let mut report = description();
    let tied = &mut report.allocations[0];
    tied.uses.push(ResourceUse {
        owner: id("load-1", "output-projection"),
        role: ResourceRole::Parameters,
    });
    let mut converted = tied.clone();
    converted.identity.key = "embedding-f32-conversion".into();
    let mut replicated = tied.clone();
    replicated.identity.key = "embedding-replica".into();
    replicated.placement = Observed::exact(id("host-1", "independent-device-1"), "registry");
    let mut independently_loaded = tied.clone();
    independently_loaded.identity.scope = "load-2".into();
    for usage in &mut independently_loaded.uses {
        usage.owner.scope = "load-2".into();
    }
    report
        .allocations
        .extend([converted, replicated, independently_loaded]);
    report.validate().unwrap();

    assert_eq!(report.allocations[0].uses.len(), 2);
    assert_eq!(report.allocations.len(), 4);
    assert_ne!(
        report.allocations[0].identity,
        report.allocations[3].identity
    );
    // Conversion/independent load share capacity with the original, not backing.
    assert_eq!(
        report.allocations[0].placement,
        report.allocations[1].placement
    );
    assert_eq!(
        report.allocations[0].placement,
        report.allocations[3].placement
    );
    assert_ne!(
        report.allocations[0].placement,
        report.allocations[2].placement
    );

    // A second name/use for one allocation must be attached to its uses, not
    // emitted as another allocation that a future composer might double charge.
    report.allocations.push(report.allocations[0].clone());
    assert!(report.validate().is_err());
}

#[test]
fn pool_identity_distinguishes_hosts_and_can_express_unified_aliases() {
    let mut report = description();
    let retained = allocation(
        "retained-features",
        "prediction",
        ResourceRole::RetainedTensor,
    );
    assert_eq!(retained.placement, report.allocations[0].placement);
    report.allocations.push(retained);
    let mut remote = allocation("remote-state", "state", ResourceRole::MutableState);
    remote.placement = Observed::Available {
        value: id("host-2", "shared-dram"),
        kind: ObservationKind::Observational,
        source: "current remote pool registry".into(),
    };
    assert_ne!(remote.placement, report.allocations[0].placement);
    report.allocations.push(remote);
    report.validate().unwrap();
}

#[test]
fn evaluated_bounds_express_nonlinear_scratch_and_rounded_horizon_capacity() {
    let mut report = description();
    report.context.dimensions =
        BTreeMap::from([("cached_positions".into(), 39), ("query_rows".into(), 1)]);
    report.horizon.dimensions = BTreeMap::from([
        ("additional_positions".into(), 30),
        ("max_query_rows".into(), 8),
    ]);
    let mut cache = allocation("cache", "layer-0", ResourceRole::MutableState);
    cache.size = ResourceSize::ContextDependent {
        current: extent(39 * 32, 64 * 32),
        horizon_peak: extent(69 * 32, 128 * 32),
    };
    // Producer evaluated query-by-key geometry and native capacity. Workspace
    // may be zero again at the endpoint; the interior peak must still be listed.
    let mut score = allocation("score-matrix", "attention", ResourceRole::Workspace);
    score.size = ResourceSize::ContextDependent {
        current: extent(0, 0),
        horizon_peak: extent(8 * 69 * 4, 4096),
    };
    report.allocations.extend([cache, score]);
    report.validate().unwrap();
    let decoded: ResourceDescription =
        serde_json::from_str(&serde_json::to_string(&report).unwrap()).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded, report);
}

#[test]
fn unknown_capacity_placement_and_coverage_survive_serialization() {
    let mut report = description();
    report.coverage = ResourceCoverage::Partial {
        reasons: vec!["native projection scratch is not described".into()],
    };
    let unknown = ResourceExtent {
        payload: ResourceByteBounds::exact(1024),
        capacity: ResourceByteBounds::unknown(1024, "allocation alignment unavailable"),
    };
    report.allocations[0].placement = Observed::unavailable("pool registry unavailable");
    report.allocations[0].size = ResourceSize::ContextDependent {
        current: unknown.clone(),
        horizon_peak: unknown,
    };
    report.validate().unwrap();
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(
        json["allocations"][0]["size"]["current"]["capacity"]["upper_bytes"],
        serde_json::Value::Null
    );
    let decoded: ResourceDescription = serde_json::from_value(json).unwrap();
    assert_eq!(decoded, report);
    decoded.validate().unwrap();

    // No entries does not cure missing coverage.
    report.allocations.clear();
    report.validate().unwrap();
    assert!(matches!(report.coverage, ResourceCoverage::Partial { .. }));
}

#[test]
fn missing_coverage_never_becomes_complete() {
    let mut json = serde_json::to_value(description()).unwrap();
    json.as_object_mut().unwrap().remove("coverage");
    json["allocations"] = serde_json::json!([]);
    let decoded: ResourceDescription = serde_json::from_value(json).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded.coverage, ResourceCoverage::Unspecified);

    let mut intentionally_empty = description();
    intentionally_empty.allocations.clear();
    intentionally_empty.validate().unwrap();
    assert_eq!(intentionally_empty.coverage, ResourceCoverage::Complete);
}

#[test]
fn invalid_decoded_bounds_and_horizons_are_rejected() {
    let mut report = description();
    report.allocations[0].size = ResourceSize::ContextDependent {
        current: extent(39 * 32, 64 * 32),
        horizon_peak: extent(70 * 32, 128 * 32),
    };
    let original = serde_json::to_value(report).unwrap();
    for (field, bad_value) in [
        ("lower_bytes", serde_json::json!(5000)),
        ("upper_bytes", serde_json::json!(0)),
        ("upper_bytes", serde_json::Value::Null),
        ("detail", serde_json::json!(" ")),
    ] {
        let mut json = original.clone();
        json["allocations"][0]["size"]["current"]["payload"][field] = bad_value;
        let decoded: ResourceDescription = serde_json::from_value(json).unwrap();
        assert!(decoded.validate().is_err(), "invalid {field} accepted");
    }

    for bad_peak in [extent(0, 4096), extent(2208, 1024)] {
        let mut json = original.clone();
        json["allocations"][0]["size"]["horizon_peak"] = serde_json::to_value(bad_peak).unwrap();
        let decoded: ResourceDescription = serde_json::from_value(json).unwrap();
        assert!(decoded.validate().is_err());
    }

    let unknown_start = ResourceSize::ContextDependent {
        current: ResourceExtent {
            payload: ResourceByteBounds::exact(1024),
            capacity: ResourceByteBounds::unknown(1024, "native capacity unknown"),
        },
        horizon_peak: extent(1024, 4096),
    };
    assert!(unknown_start.validate().is_err());
}

#[test]
fn invalid_identities_coverage_and_guessed_pool_placement_are_rejected() {
    let mut report = description();
    report.allocations[0].uses.clear();
    assert!(report.validate().is_err());
    report = description();
    let repeated_use = report.allocations[0].uses[0].clone();
    report.allocations[0].uses.push(repeated_use);
    assert!(report.validate().is_err());
    report = description();
    report.scope.scope.clear();
    assert!(report.validate().is_err());
    report = description();
    report.context.dimensions.insert(" ".into(), 1);
    assert!(report.validate().is_err());
    report = description();
    report.coverage = ResourceCoverage::Partial { reasons: vec![] };
    assert!(report.validate().is_err());
    report.coverage = ResourceCoverage::Partial {
        reasons: vec!["".into()],
    };
    assert!(report.validate().is_err());
    report = description();
    report.allocations[0].placement = Observed::Available {
        value: id("host-1", "guessed-unified-pool"),
        kind: ObservationKind::Estimated,
        source: "device probably uses unified memory".into(),
    };
    assert!(report.validate().is_err());
    report = description();
    report.schema_version += 1;
    assert_eq!(
        report.validate(),
        Err(ResourceDescriptionError::UnsupportedVersion(2))
    );
}

#[test]
fn maximum_representable_capacity_is_valid_without_wrapping_or_summing() {
    let mut report = description();
    report.allocations[0].size = ResourceSize::ContextDependent {
        current: extent(u64::MAX - 1, u64::MAX),
        horizon_peak: extent(u64::MAX, u64::MAX),
    };
    report.validate().unwrap();
    let mut impossible = extent(u64::MAX, u64::MAX - 1);
    assert!(impossible.validate().is_err());
    impossible.capacity = ResourceByteBounds::unknown(u64::MAX, "unbounded allocation");
    impossible.validate().unwrap();
}

#[test]
fn new_wire_contract_uses_snake_case_and_explicit_tagged_records() {
    let json = serde_json::to_value(description()).unwrap();
    assert_eq!(json["schema_version"], 1);
    assert_eq!(json["coverage"], serde_json::json!({ "kind": "complete" }));
    assert_eq!(json["allocations"][0]["uses"][0]["role"], "parameters");
    assert_eq!(json["allocations"][0]["size"]["kind"], "fixed");
    assert_eq!(json["allocations"][0]["placement"]["status"], "available");
    assert_eq!(
        serde_json::to_value(ResourceRole::MutableState).unwrap(),
        "mutable_state"
    );
    assert_eq!(
        serde_json::to_value(ResourceRole::RetainedTensor).unwrap(),
        "retained_tensor"
    );
}

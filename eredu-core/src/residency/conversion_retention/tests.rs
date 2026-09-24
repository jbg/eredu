use super::*;
use crate::residency::{OffloadReport, OffloadTelemetry};
use serde_json::json;

fn group(scope: &str) -> ParameterConversionRetentionGroup {
    ParameterConversionRetentionGroup(ResourceIdentity {
        scope: scope.into(),
        key: "conversion_retention".into(),
    })
}

fn usage(claims: u64, payload: u64, reserved: u64) -> ParameterConversionRetentionUsage {
    ParameterConversionRetentionUsage {
        retained_claims: claims,
        retained_payload_bytes: payload,
        reserved_payload_bytes: reserved,
        retained_backing_capacity_bytes: Observed::unsupported("native capacity is not observable"),
    }
}

fn report(scope: &str) -> ParameterConversionRetentionReport {
    ParameterConversionRetentionReport {
        group: group(scope),
        policy: Observed::exact(
            ParameterConversionRetentionPolicyReport::resolve(
                None,
                ParameterConversionRetentionEligibility::Eligible,
            ),
            "selected execution",
        ),
        usage: Observed::exact(usage(2, 1024, 512), "retention controller snapshot"),
    }
}

#[test]
fn policy_normalization_preserves_request_and_provenance() {
    use ParameterConversionRetentionPolicy::{Bounded, Disabled, Unlimited};
    for (requested, effective) in [
        (Disabled, Disabled),
        (Bounded { max_bytes: 0 }, Disabled),
        (Bounded { max_bytes: 1 }, Bounded { max_bytes: 1 }),
        (
            Bounded {
                max_bytes: u64::MAX,
            },
            Bounded {
                max_bytes: u64::MAX,
            },
        ),
        (Unlimited, Unlimited),
    ] {
        let selected = ParameterConversionRetentionPolicyReport::resolve(
            Some(requested),
            ParameterConversionRetentionEligibility::Eligible,
        );
        assert_eq!(selected.requested, requested);
        assert_eq!(selected.effective, effective);
        assert_eq!(
            selected.source,
            ParameterConversionRetentionPolicySource::Explicit
        );
        let encoded = serde_json::to_value(&selected).unwrap();
        assert_eq!(encoded["source"], "explicit");
        assert_eq!(
            serde_json::from_value::<ParameterConversionRetentionPolicyReport>(encoded).unwrap(),
            selected
        );
    }
    let managed = ParameterConversionRetentionPolicyReport::resolve(
        None,
        ParameterConversionRetentionEligibility::Eligible,
    );
    assert_eq!(
        managed.requested,
        Bounded {
            max_bytes: 268_435_456
        }
    );
    assert_eq!(managed.effective, managed.requested);
    assert_eq!(
        managed.source,
        ParameterConversionRetentionPolicySource::ManagedDefault
    );
    assert_eq!(
        serde_json::to_value(&managed).unwrap()["effective"],
        json!({"kind": "bounded", "max_bytes": 268_435_456})
    );
}

#[test]
fn exclusions_disable_effective_policy_without_hiding_the_request() {
    use ParameterConversionRetentionEligibility::*;
    for (eligibility, kind) in [
        (HostLayerwise, "host_layerwise"),
        (DiskStreamed, "disk_streamed"),
        (DeviceResidencyLimit, "device_residency_limit"),
        (
            Unsupported {
                reason: "mechanism has no retained parameter owners".into(),
            },
            "unsupported",
        ),
    ] {
        for requested in [None, Some(ParameterConversionRetentionPolicy::Unlimited)] {
            let selected =
                ParameterConversionRetentionPolicyReport::resolve(requested, eligibility.clone());
            assert_eq!(
                selected.effective,
                ParameterConversionRetentionPolicy::Disabled
            );
            assert_eq!(selected.eligibility, eligibility);
            assert_eq!(
                selected.requested,
                requested.unwrap_or(ParameterConversionRetentionPolicy::MANAGED_DEFAULT)
            );
            let encoded = serde_json::to_value(&selected).unwrap();
            assert_eq!(encoded["eligibility"]["kind"], kind);
            assert_eq!(
                serde_json::from_value::<ParameterConversionRetentionPolicyReport>(encoded)
                    .unwrap(),
                selected
            );
        }
    }
}

#[test]
fn composed_report_round_trip_keeps_independent_group_scope_and_usage() {
    let target = report("target_instance");
    let mut drafter = report("external_drafter_instance");
    drafter.usage = Observed::exact(usage(1, 256, 0), "drafter controller");
    assert_ne!(target.group, drafter.group);
    let snapshot = OffloadTelemetry::default()
        .snapshot()
        .with_parameter_conversion_retention(Observed::exact(
            vec![target, drafter],
            "composed execution groups",
        ));
    let encoded = serde_json::to_value(&snapshot).unwrap();
    assert_eq!(
        encoded["parameter_conversion_retention"]["value"][0]["policy"]["value"]["source"],
        "managed_default"
    );
    assert_eq!(
        serde_json::from_value::<OffloadReport>(encoded).unwrap(),
        snapshot
    );
    let groups = snapshot.parameter_conversion_retention().value().unwrap();
    let target_usage = groups[0].usage.value().unwrap();
    assert_eq!(target_usage.retained_payload_bytes, 1024);
    assert_eq!(target_usage.reserved_payload_bytes, 512);
    assert!(matches!(
        target_usage.retained_backing_capacity_bytes,
        Observed::Unsupported { .. }
    ));
}

#[test]
fn missing_and_unsupported_observations_are_not_observed_empty() {
    let missing = OffloadTelemetry::default().snapshot();
    let unsupported = missing
        .clone()
        .with_parameter_conversion_retention(Observed::unsupported(
            "backend does not expose retention accounting",
        ));
    let empty = missing
        .clone()
        .with_parameter_conversion_retention(Observed::exact(
            vec![],
            "no selected retention groups",
        ));
    assert!(matches!(
        missing.parameter_conversion_retention(),
        Observed::Unavailable { .. }
    ));
    assert!(matches!(
        unsupported.parameter_conversion_retention(),
        Observed::Unsupported { .. }
    ));
    assert_eq!(
        empty.parameter_conversion_retention().value(),
        Some(&vec![])
    );
    assert_ne!(empty, missing);
    assert_ne!(empty, unsupported);
    for snapshot in [missing, unsupported, empty] {
        let encoded = serde_json::to_string(&snapshot).unwrap();
        assert_eq!(
            serde_json::from_str::<OffloadReport>(&encoded).unwrap(),
            snapshot
        );
    }

    let mut group_report = report("target");
    group_report.usage = Observed::exact(usage(0, 0, 0), "observed empty cache");
    let zero = group_report.usage.clone();
    group_report.usage = Observed::unsupported("usage observation unsupported");
    assert_ne!(group_report.usage, zero);
    let encoded = serde_json::to_string(&group_report).unwrap();
    assert_eq!(
        serde_json::from_str::<ParameterConversionRetentionReport>(&encoded).unwrap(),
        group_report
    );
}

#[test]
fn old_offload_record_preserves_accounting_without_inventing_policy() {
    let mut telemetry = OffloadTelemetry::default();
    telemetry.record_eviction(4096);
    let original = telemetry.snapshot();
    let mut legacy = serde_json::to_value(&original).unwrap();
    legacy
        .as_object_mut()
        .unwrap()
        .remove("parameter_conversion_retention");
    let decoded: OffloadReport = serde_json::from_value(legacy).unwrap();
    assert_eq!(decoded, original);
    assert_eq!(decoded.evictions().bytes, 4096);
    assert!(matches!(
        decoded.parameter_conversion_retention(),
        Observed::Unavailable { .. }
    ));
}

#[test]
fn omitted_policy_usage_and_native_facts_remain_unknown() {
    let old: ParameterConversionRetentionReport = serde_json::from_value(json!({
        "group": { "scope": "target", "key": "conversion_retention" }
    }))
    .unwrap();
    assert!(matches!(old.policy, Observed::Unavailable { .. }));
    assert!(matches!(old.usage, Observed::Unavailable { .. }));

    let old_usage: ParameterConversionRetentionUsage = serde_json::from_value(json!({
        "retained_claims": 1,
        "retained_payload_bytes": 4096,
        "reserved_payload_bytes": 128
    }))
    .unwrap();
    assert_eq!(old_usage.retained_payload_bytes, 4096);
    assert_eq!(old_usage.reserved_payload_bytes, 128);
    assert!(matches!(
        old_usage.retained_backing_capacity_bytes,
        Observed::Unavailable { .. }
    ));
}

#[test]
fn trim_claim_release_is_independent_of_backing_reclamation() {
    for reclaimed_backing_bytes in [
        Observed::unsupported("native live backing cannot be measured"),
        Observed::exact(0, "another group retains the backing"),
        Observed::exact(8192, "native allocator received the backing"),
    ] {
        let trim = ParameterConversionRetentionTrimReport {
            group: group("target"),
            released_claims: 2,
            released_payload_bytes: 4096,
            remaining: usage(1, 256, 0),
            reclaimed_backing_bytes,
        };
        let encoded = serde_json::to_value(&trim).unwrap();
        assert_eq!(
            serde_json::from_value::<ParameterConversionRetentionTrimReport>(encoded.clone())
                .unwrap(),
            trim
        );
        let mut legacy = encoded;
        legacy
            .as_object_mut()
            .unwrap()
            .remove("reclaimed_backing_bytes");
        let decoded: ParameterConversionRetentionTrimReport =
            serde_json::from_value(legacy).unwrap();
        assert_eq!(decoded.released_payload_bytes, 4096);
        assert_eq!(decoded.remaining.retained_claims, 1);
        assert!(matches!(
            decoded.reclaimed_backing_bytes,
            Observed::Unavailable { .. }
        ));
    }
}

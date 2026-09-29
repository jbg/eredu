use super::*;
use crate::resource_lifetimes::ResourceLifetime;
use eredu_core::{resources::*, Observed};
use std::collections::BTreeMap;

fn id(key: &str) -> ResourceIdentity {
    ResourceIdentity {
        scope: "fixture".into(),
        key: key.into(),
    }
}
fn description(bytes: u64) -> ResourceLifetimeDescription {
    ResourceLifetimeDescription {
        resources: ResourceDescription {
            schema_version: RESOURCE_DESCRIPTION_SCHEMA_VERSION,
            scope: id("transfer"),
            context: ResourceContext::default(),
            horizon: ResourceContext::default(),
            coverage: ResourceCoverage::Partial {
                reasons: vec!["device storage is not described".into()],
            },
            allocations: vec![ResourceAllocation {
                identity: id("buffer"),
                uses: vec![ResourceUse {
                    owner: id("transfer"),
                    role: ResourceRole::Parameters,
                }],
                placement: Observed::unavailable("unbound physical pool"),
                size: ResourceSize::Fixed {
                    extent: ResourceExtent {
                        payload: ResourceByteBounds::exact(bytes),
                        capacity: ResourceByteBounds::exact(bytes + 8),
                    },
                },
            }],
        },
        lifetimes: BTreeMap::from([(id("buffer"), vec![ResourceLifetime::Owner(id("transfer"))])]),
        live_at_acquire: true,
    }
}

#[test]
fn detached_inventory_preserves_unknown_nodes_limits_and_shared_holds() {
    let limits = DetachedResourceLimits::new(3, 2).unwrap();
    let inventory = DetachedResourceInventory::new(
        "test queue".into(),
        limits,
        3,
        1,
        DetachedResourceScan::Truncated,
        vec![description(16), description(16)],
    )
    .unwrap();
    assert_eq!(inventory.limits(), limits);
    assert_eq!(inventory.inspected_nodes(), 3);
    assert_eq!(inventory.undescribed_nodes(), 1);
    assert_eq!(inventory.scan(), &DetachedResourceScan::Truncated);
    assert_eq!(
        inventory.descriptions().len(),
        2,
        "alias records retain separate references"
    );
    assert_eq!(inventory.descriptions()[0], description(16));
    // A complete traversal can still have unknown resource ownership.
    let unknown = DetachedResourceInventory::new(
        "test queue".into(),
        limits,
        1,
        1,
        DetachedResourceScan::Complete,
        vec![],
    )
    .unwrap();
    assert_eq!(unknown.undescribed_nodes(), 1);
    let unavailable = DetachedResourceInventory::new(
        "test queue".into(),
        limits,
        0,
        0,
        DetachedResourceScan::Unavailable {
            reason: "busy".into(),
        },
        vec![],
    )
    .unwrap();
    assert!(matches!(
        unavailable.scan(),
        DetachedResourceScan::Unavailable { .. }
    ));
}

#[test]
fn detached_inventory_rejects_malformed_counts_bounds_and_conflicting_backings() {
    assert!(DetachedResourceLimits::new(0, 1).is_err());
    assert!(DetachedResourceLimits::new(1, 0).is_err());
    let limits = DetachedResourceLimits::new(2, 1).unwrap();
    let make = |source: &str, n, unknown, scan, descriptions| {
        DetachedResourceInventory::new(source.into(), limits, n, unknown, scan, descriptions)
    };
    assert!(make(" ", 0, 0, DetachedResourceScan::Complete, vec![]).is_err());
    assert!(make("queue", 3, 3, DetachedResourceScan::Complete, vec![]).is_err());
    assert!(make("queue", 1, 0, DetachedResourceScan::Complete, vec![]).is_err());
    assert!(make(
        "queue",
        1,
        usize::MAX,
        DetachedResourceScan::Complete,
        vec![description(16)]
    )
    .is_err());
    assert!(make(
        "queue",
        1,
        1,
        DetachedResourceScan::Unavailable {
            reason: "busy".into()
        },
        vec![]
    )
    .is_err());
    assert!(make(
        "queue",
        0,
        0,
        DetachedResourceScan::Unavailable { reason: "".into() },
        vec![]
    )
    .is_err());
    assert!(make(
        "queue",
        2,
        0,
        DetachedResourceScan::Complete,
        vec![description(16), description(16)]
    )
    .is_err());
    assert!(DetachedResourceInventory::new(
        "queue".into(),
        DetachedResourceLimits::new(2, 2).unwrap(),
        2,
        0,
        DetachedResourceScan::Complete,
        vec![description(16), description(32)]
    )
    .is_err());
    let mut malformed = description(16);
    malformed.lifetimes.insert(
        id("absent-buffer"),
        vec![ResourceLifetime::Owner(id("transfer"))],
    );
    assert!(make(
        "queue",
        1,
        0,
        DetachedResourceScan::Complete,
        vec![malformed]
    )
    .is_err());
}

//! Public lifetime composition checked against explicit physical live sets.
use std::collections::{BTreeMap, BTreeSet};

use eredu_core::{resources::*, Observed};
use eredu_runtime::resource_lifetimes::*;

fn id(key: impl Into<String>) -> ResourceIdentity {
    ResourceIdentity {
        scope: "synthetic-execution".into(),
        key: key.into(),
    }
}

fn acquire(index: usize, backing: usize, pool: usize) -> ResourceLifetimeEvent {
    let identity = id(format!("allocation-{backing}"));
    let token = id(format!("boundary-{index}"));
    let lifetime = match index % 3 {
        0 => ResourceLifetime::NativeCompletion(token),
        1 => ResourceLifetime::Evaluation(token),
        _ => ResourceLifetime::Owner(token),
    };
    let payload = 7 + 11 * backing as u64;
    let extent = ResourceExtent {
        payload: ResourceByteBounds::exact(payload),
        capacity: ResourceByteBounds::exact(payload + 5),
    };
    ResourceLifetimeEvent::Acquire(ResourceLifetimeDescription {
        resources: ResourceDescription {
            schema_version: RESOURCE_DESCRIPTION_SCHEMA_VERSION,
            scope: id(format!("invocation-{index}")),
            context: ResourceContext::default(),
            horizon: ResourceContext::default(),
            coverage: ResourceCoverage::Complete,
            allocations: vec![ResourceAllocation {
                identity: identity.clone(),
                uses: vec![ResourceUse {
                    owner: id(format!("owner-{index}")),
                    role: ResourceRole::Workspace,
                }],
                placement: Observed::exact(id(format!("pool-{pool}")), "synthetic placement"),
                size: ResourceSize::Fixed { extent },
            }],
        },
        lifetimes: BTreeMap::from([(identity, vec![lifetime])]),
        live_at_acquire: true,
    })
}

fn release(index: usize) -> ResourceLifetimeEvent {
    let token = id(format!("boundary-{index}"));
    match index % 3 {
        0 => ResourceLifetimeEvent::Complete(token),
        1 => ResourceLifetimeEvent::Evaluate(token),
        _ => ResourceLifetimeEvent::Release(token),
    }
}

#[test]
fn exact_peaks_match_exhaustive_small_physical_live_sets() {
    // Every nonempty interval on three ticks, including nesting, crossing,
    // adjacency and complete overlap. The oracle uses only set membership.
    let intervals = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    for first in intervals {
        for second in intervals {
            for third in intervals {
                let lifetimes = [first, second, third];
                for sharing in [false, true] {
                    for unified in [false, true] {
                        let backings = [0, if sharing { 0 } else { 1 }, 2];
                        let pools = backings.map(|backing| if unified { 0 } else { backing % 2 });
                        let mut events = Vec::new();
                        let mut expected = BTreeMap::<usize, (u64, u64)>::new();
                        for tick in 0..=3 {
                            for (index, (_, end)) in lifetimes.iter().enumerate() {
                                if *end == tick {
                                    events.push(release(index));
                                }
                            }
                            for (index, (start, _)) in lifetimes.iter().enumerate() {
                                if *start == tick {
                                    events.push(acquire(index, backings[index], pools[index]));
                                }
                            }
                            let live = lifetimes
                                .iter()
                                .enumerate()
                                .filter(|(_, (start, end))| *start <= tick && tick < *end)
                                .map(|(index, _)| (pools[index], backings[index]))
                                .collect::<BTreeSet<_>>();
                            let mut current = BTreeMap::<usize, (u64, u64)>::new();
                            for (pool, backing) in live {
                                let payload = 7 + 11 * backing as u64;
                                let bytes = current.entry(pool).or_default();
                                bytes.0 += payload;
                                bytes.1 += payload + 5;
                            }
                            for (pool, bytes) in current {
                                let peak = expected.entry(pool).or_default();
                                peak.0 = peak.0.max(bytes.0);
                                peak.1 = peak.1.max(bytes.1);
                            }
                        }
                        let report = compose_resource_peaks(&ResourceLifetimePlan {
                            coverage: ResourceCoverage::Complete,
                            events,
                        })
                        .unwrap();
                        assert!(report.missing.is_empty());
                        assert!(report.unplaced.is_empty());
                        assert_eq!(report.pools.len(), expected.len());
                        for (pool, (payload, capacity)) in expected {
                            let observed = &report
                                .pools
                                .iter()
                                .find(|entry| entry.pool == id(format!("pool-{pool}")))
                                .unwrap()
                                .peak;
                            assert_eq!(
                                (observed.payload.lower_bytes, observed.payload.upper_bytes),
                                (payload, Some(payload)),
                                "{lifetimes:?}, sharing={sharing}, unified={unified}, pool={pool}"
                            );
                            assert_eq!(
                                (observed.capacity.lower_bytes, observed.capacity.upper_bytes),
                                (capacity, Some(capacity))
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn release_kinds_have_distinct_namespaces_even_with_identical_labels() {
    let mut first = acquire(0, 0, 0);
    let mut second = acquire(1, 1, 0);
    let shared_label = id("one-label");
    for (event, lifetime) in [
        (
            &mut first,
            ResourceLifetime::NativeCompletion(shared_label.clone()),
        ),
        (
            &mut second,
            ResourceLifetime::Evaluation(shared_label.clone()),
        ),
    ] {
        let ResourceLifetimeEvent::Acquire(description) = event else {
            unreachable!()
        };
        *description.lifetimes.values_mut().next().unwrap() = vec![lifetime];
    }
    let report = compose_resource_peaks(&ResourceLifetimePlan {
        coverage: ResourceCoverage::Complete,
        events: vec![
            first,
            second,
            ResourceLifetimeEvent::Complete(shared_label.clone()),
            acquire(2, 2, 0),
            ResourceLifetimeEvent::Evaluate(shared_label),
            release(2),
        ],
    })
    .unwrap();
    // The 18-byte evaluation allocation remains after native completion,
    // overlapping the new 29-byte owner-held allocation.
    assert_eq!(
        report.pools[0].peak.payload,
        ResourceByteBounds {
            lower_bytes: 47,
            upper_bytes: Some(47),
            kind: eredu_core::ObservationKind::Exact,
            detail: report.pools[0].peak.payload.detail.clone(),
        }
    );
}

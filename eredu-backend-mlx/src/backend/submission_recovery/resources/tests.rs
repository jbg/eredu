use super::*;
use eredu_core::{resources::*, Observed};
use eredu_runtime::resource_lifetimes::ResourceLifetime;
use std::{cell::Cell, collections::BTreeMap};

struct Gate {
    done: Rc<Cell<bool>>,
    polls: Rc<Cell<usize>>,
}
impl Probe for Gate {
    fn seal(&mut self) {}
    fn progress(&self) -> Status {
        self.polls.set(self.polls.get() + 1);
        Status {
            settled: self.done.get(),
            failed: !self.done.get(),
            blocked: false,
        }
    }
}
struct Held {
    described: bool,
    drops: Rc<Cell<usize>>,
    observations: Rc<Cell<usize>>,
}
impl Retention for Held {
    fn observe(&self, _: Status) {
        self.observations.set(self.observations.get() + 1);
    }
    fn describe_resources(
        &self,
        maximum: usize,
    ) -> Result<Option<ResourceLifetimeDescription>, ResourceDescriptionError> {
        if !self.described || maximum == 0 {
            return Ok(None);
        }
        Ok(Some(description()))
    }
}
impl Drop for Held {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}
fn id(key: &str) -> ResourceIdentity {
    ResourceIdentity {
        scope: "recovery.fixture".into(),
        key: key.into(),
    }
}
fn description() -> ResourceLifetimeDescription {
    ResourceLifetimeDescription {
        resources: ResourceDescription {
            schema_version: RESOURCE_DESCRIPTION_SCHEMA_VERSION,
            scope: id("transfer"),
            context: ResourceContext::default(),
            horizon: ResourceContext::default(),
            coverage: ResourceCoverage::Partial {
                reasons: vec!["fixture host sources only".into()],
            },
            allocations: vec![ResourceAllocation {
                identity: id("buffer"),
                uses: vec![ResourceUse {
                    owner: id("transfer"),
                    role: ResourceRole::Parameters,
                }],
                placement: Observed::unavailable("unknown pool"),
                size: ResourceSize::Fixed {
                    extent: ResourceExtent {
                        payload: ResourceByteBounds::exact(12),
                        capacity: ResourceByteBounds::exact(16),
                    },
                },
            }],
        },
        lifetimes: BTreeMap::from([(id("buffer"), vec![ResourceLifetime::Owner(id("transfer"))])]),
        live_at_acquire: true,
    }
}

#[test]
fn detached_inventory_is_bounded_and_does_not_poll_or_release() {
    let done = Rc::new(Cell::new(false));
    let polls = Rc::new(Cell::new(0));
    let observations = Rc::new(Cell::new(0));
    let drops = Rc::new(Cell::new(0));
    for described in [true, true, false] {
        let recovery = Recovery::with_probe(
            Held {
                described,
                drops: drops.clone(),
                observations: observations.clone(),
            },
            Gate {
                done: done.clone(),
                polls: polls.clone(),
            },
        );
        let failed = recovery.finish();
        assert!(failed.failed && !failed.settled);
    }
    let before = (polls.get(), observations.get());
    let truncated = describe_resources(DetachedResourceLimits::new(2, 1).unwrap()).unwrap();
    assert_eq!(truncated.inspected_nodes(), 2);
    assert_eq!(truncated.undescribed_nodes(), 1);
    assert_eq!(truncated.descriptions().len(), 1);
    assert_eq!(truncated.scan(), &DetachedResourceScan::Truncated);
    let scarce = describe_resources(DetachedResourceLimits::new(3, 1).unwrap()).unwrap();
    assert_eq!(scarce.scan(), &DetachedResourceScan::Complete);
    assert_eq!(scarce.undescribed_nodes(), 2);
    let complete = describe_resources(DetachedResourceLimits::new(3, 2).unwrap()).unwrap();
    assert_eq!(complete.inspected_nodes(), 3);
    assert_eq!(complete.undescribed_nodes(), 1);
    assert_eq!(complete.descriptions(), &[description(), description()]);
    assert_eq!((polls.get(), observations.get()), before);
    assert_eq!(drops.get(), 0);
    // Even terminal evidence cannot make observation itself retire the node.
    done.set(true);
    assert_eq!(
        describe_resources(DetachedResourceLimits::new(3, 2).unwrap()).unwrap(),
        complete
    );
    assert_eq!(drops.get(), 0);
    wait_for_retirement(|| drops.get() == 3);
    let empty = describe_resources(DetachedResourceLimits::new(3, 2).unwrap()).unwrap();
    assert_eq!(empty.inspected_nodes(), 0);
    assert_eq!(empty.scan(), &DetachedResourceScan::Complete);
}

#[test]
fn detached_inventory_reports_borrow_contention_without_reaping() {
    ORPHANS.with(|orphans| {
        let _exclusive = orphans.borrow_mut();
        let snapshot = describe_resources(DetachedResourceLimits::new(1, 1).unwrap()).unwrap();
        assert!(matches!(
            snapshot.scan(),
            DetachedResourceScan::Unavailable { .. }
        ));
        assert_eq!(snapshot.inspected_nodes(), 0);
        assert!(snapshot.descriptions().is_empty());
    });
}

#[test]
fn detached_inventory_reports_in_progress_retirement_and_recursive_observation() {
    struct Reentrant;
    impl Retention for Reentrant {
        fn observe(&self, _: Status) {
            if REAPING.with(|flag| flag.get()) {
                assert!(matches!(
                    describe_resources(DetachedResourceLimits::new(1, 1).unwrap())
                        .unwrap()
                        .scan(),
                    DetachedResourceScan::Unavailable { .. }
                ));
            }
        }
        fn describe_resources(
            &self,
            _: usize,
        ) -> Result<Option<ResourceLifetimeDescription>, ResourceDescriptionError> {
            assert!(matches!(
                describe_resources(DetachedResourceLimits::new(1, 1).unwrap())?.scan(),
                DetachedResourceScan::Unavailable { .. }
            ));
            Ok(None)
        }
    }
    let done = Rc::new(Cell::new(false));
    drop(Recovery::with_probe(
        Reentrant,
        Gate {
            done: done.clone(),
            polls: Rc::new(Cell::new(0)),
        },
    ));
    let snapshot = describe_resources(DetachedResourceLimits::new(1, 1).unwrap()).unwrap();
    assert_eq!(snapshot.inspected_nodes(), 1);
    assert_eq!(snapshot.undescribed_nodes(), 1);
    reap();
    done.set(true);
    wait_for_retirement(|| {
        describe_resources(DetachedResourceLimits::new(1, 1).unwrap())
            .unwrap()
            .inspected_nodes()
            == 0
    });
}

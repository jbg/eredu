//! Backend-independent completion ownership for distributed operations.
//!
//! Distributed MLX primitives are lazy just like ordinary array operations.
//! This module couples a submitted result with the exact `safemlx` event which
//! completes it, so pipeline and collective callers do not need to duplicate
//! `eval` plus whole-stream synchronization sequences.

use safemlx::{transforms::async_eval_with_event, Array, Event, EventBackend, Stream};
#[cfg(test)]
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use crate::backend::{
    error::Error,
    runtime::distributed::{topology::CommunicationRouteRealization, Group},
    submission_recovery::{Probe, Recovery, Retention, Status},
};

#[derive(Debug)]
struct NativeResources {
    event: RefCell<Option<Rc<Event>>>,
    original_event: RefCell<Option<Rc<safemlx::OperationEvent>>>,
    housekeeping: RefCell<Option<safemlx::RegisteredThreadRuntimeHousekeeping>>,
    arrays: Vec<Array>,
    _count_buffers: Vec<Vec<usize>>,
    groups: Vec<Group>,
    _routes: Vec<CommunicationRouteRealization>,
    _streams: Vec<Stream>,
    status: Cell<Status>,
    host_failed: Cell<bool>,
    children: Cell<usize>,
    child_failed: Cell<bool>,
}

/// Each strong native alias retains only account/source custody after its Rc.
/// This releases the actual Rc allocation before the final funding alias.
#[derive(Debug)]
struct NativeOwner {
    resources: Option<Rc<NativeResources>>,
    custody: Option<prepared::ResourceCustody>,
    ordinary: Option<crate::backend::nn::shared::OrdinaryExecutionOwner>,
}
impl Clone for NativeOwner {
    fn clone(&self) -> Self { Self { resources: self.resources.clone(), custody: self.custody.clone(), ordinary: self.ordinary.clone() } }
}
impl std::ops::Deref for NativeOwner {
    type Target = NativeResources;
    fn deref(&self) -> &Self::Target { self.resources.as_deref().expect("native resource owner") }
}
impl Drop for NativeOwner {
    fn drop(&mut self) {
        drop(self.resources.take());
        // Retire dead weak shells while this alias still retains funding. A
        // reentrant list loan merely defers this pruning to its next use.
        let _ = NATIVE_RESOURCE_OWNERS.try_with(|owners| {
            if let Ok(mut owners) = owners.try_borrow_mut() {
                owners.retain(|owner| owner.strong_count() != 0);
            }
        });
    }
}

impl Retention for NativeOwner {
    fn observe(&self, status: Status) {
        self.status.set(status);
    }
}

impl NativeResources {
    fn new(
        arrays: Vec<Array>,
        count_buffers: Vec<Vec<usize>>,
        groups: Vec<Group>,
        routes: Vec<CommunicationRouteRealization>,
        streams: Vec<Stream>,
    ) -> NativeOwner {
        Self::from_owned(arrays, count_buffers, groups, routes, streams, None, None)
    }

    fn from_owned(
        arrays: Vec<Array>, count_buffers: Vec<Vec<usize>>, groups: Vec<Group>,
        routes: Vec<CommunicationRouteRealization>, streams: Vec<Stream>,
        custody: Option<prepared::ResourceCustody>,
        destination: Option<destinations::Destination<std::rc::Weak<NativeResources>>>,
    ) -> NativeOwner {
        let resources = Rc::new(Self {
            event: RefCell::new(None),
            original_event: RefCell::new(None),
            housekeeping: RefCell::new(None),
            arrays,
            _count_buffers: count_buffers,
            groups,
            _routes: routes,
            _streams: streams,
            status: Cell::new(Status {
                settled: false,
                failed: false,
                blocked: false,
            }),
            host_failed: Cell::new(false),
            children: Cell::new(0),
            child_failed: Cell::new(false),
        });
        if !resources.groups.is_empty() || destination.is_some() {
            // Reserve registration before submission, not on the failure path.
            NATIVE_RESOURCE_OWNERS.with(|owners| {
                let mut owners = owners.borrow_mut();
                owners.retain(|owner| owner.strong_count() != 0);
                let weak = Rc::downgrade(&resources);
                match destination {
                    Some(destination) => owners.push_prepared(weak, destination),
                    None => owners.push(weak),
                }
            });
        }
        NativeOwner { resources: Some(resources), custody, ordinary: None }
    }

    fn unavailable(&self) -> bool {
        let status = self.status.get();
        (!status.settled || self.children.get() != 0)
            && (self.host_failed.get()
                || self.child_failed.get()
                || status.failed
                || status.blocked)
    }
}

/// A child observation must not replace the original submission's status.
struct NativeChildResources {
    _arrays: Vec<Array>,
    _stream: Option<Stream>,
    owner: NativeOwner,
}

impl Retention for NativeChildResources {
    fn observe(&self, status: Status) {
        if status.failed || status.blocked {
            self.owner.child_failed.set(true);
        }
    }
}

impl Drop for NativeChildResources {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.owner.host_failed.set(true);
        }
        self.owner.children.set(self.owner.children.get() - 1);
    }
}

struct ChildUnwind<'a>(&'a Cell<bool>);
impl Drop for ChildUnwind<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.set(true);
        }
    }
}

fn observe_native_child<T>(
    owner: &NativeOwner,
    arrays: Vec<Array>,
    stream: Option<Stream>,
    operation: impl FnOnce() -> safemlx::error::Result<T>,
) -> safemlx::error::Result<(T, bool)> {
    owner.children.set(
        owner
            .children
            .get()
            .checked_add(1)
            .expect("native child ticket overflow"),
    );
    let mut recovery = Recovery::begin(NativeChildResources {
        owner: owner.clone(),
        _arrays: arrays,
        _stream: stream,
    })?;
    let _unwind = ChildUnwind(&owner.host_failed);
    let result = operation();
    if result.is_err() {
        owner.host_failed.set(true);
    }
    recovery.seal();
    let status = recovery.progress();
    if status.failed || status.blocked {
        return Err(safemlx::error::Exception::custom(
            "native completion observation failed; resources remain retained",
        ));
    }
    Ok((result?, status.settled))
}

fn native_resources_releasable(recovery: &impl original::CompletionState) -> bool {
    safemlx::try_with_submission_retirement(|| {
        crate::backend::submission_recovery::reap();
        recovery.completion_status().is_ok_and(|status| status.settled) && recovery.resources().children.get() == 0
    })
    .unwrap_or(false)
}

fn check_native_status(
    recovery: &impl original::CompletionState,
) -> safemlx::error::Result<bool> {
    crate::backend::submission_recovery::reap();
    let status = recovery.completion_status()?;
    let resources = recovery.resources();
    if status.failed
        || status.blocked
        || resources.host_failed.get()
        || resources.child_failed.get()
    {
        return Err(match recovery.observer() {
            Some(observer) => observer.retained_failure().unwrap_or_else(|| observer.invalid_input_error()),
            None => match &resources.ordinary {
                Some(owner) => ordinary::failure(ordinary::Cause::Completion, owner.host()),
                None => safemlx::error::Exception::custom("native communication failed; unresolved resources remain retained"),
            },
        });
    }
    // An inline original event can complete before its enclosing model role.
    // This permits querying that exact event, never releasing the role payload.
    Ok(recovery.observer().is_some() || (status.settled && resources.children.get() == 0))
}

mod communication;
mod ordinary;
pub(crate) use ordinary::{ordinary_completed_i32_scalar, ordinary_completed_i32_scalar_control_bytes};
mod original;
mod consumers;
mod scalar;
mod readouts;
use readouts::{BoundaryHeaders, WordsResult};
pub(crate) use readouts::{PreparedCommunicationWords, PreparedCommunicationHeader, OriginalCommunicationWords, CompletedCommunicationWords,
    PreparedCommunicationU32Words,OriginalCommunicationU32Words,CompletedCommunicationU32Words};
pub(crate) use scalar::{PreparedCommunicationScalar, OriginalCommunicationBool};
use scalar::BoolResult;
use original::{CompletionRecovery, NativeEvent};
pub(crate) use original::OriginalCommunicationCompletion;
mod destinations;
pub(crate) mod prepared;
mod generic;

thread_local! {
    static NATIVE_RESOURCE_OWNERS: RefCell<destinations::Destinations<std::rc::Weak<NativeResources>>> = const { RefCell::new(destinations::Destinations::new()) };
    static DISTRIBUTED_COMPLETION_ORPHANS: RefCell<generic::DistributedCompletionOrphanQuarantine> =
        RefCell::new(generic::DistributedCompletionOrphanQuarantine::default());
    static COMMUNICATION_ORPHANS: RefCell<communication::CommunicationOrphanQuarantine> =
        RefCell::new(communication::CommunicationOrphanQuarantine::default());
    #[cfg(test)]
    static FORCE_NEXT_COMMUNICATION_PENDING: Cell<bool> = const { Cell::new(false) };
}

pub(crate) use communication::{ensure_group_available, group_source_available, group_source_controls};
#[cfg(test)]
pub(crate) use communication::{
    distributed_completion_orphan_count, force_next_communication_pending,
    release_forced_pending_orphans,
};
pub use communication::{synchronize_outputs, MlxCommunicationCompletion, MlxFailureAgreement};
mod neural;
pub use neural::MlxNeuralCommunicationCompletion;
pub use generic::DistributedCompletion;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod child_scope_tests {
    use super::*;

    struct FakeProbe(Rc<Cell<Status>>);
    impl Probe for FakeProbe {
        fn seal(&mut self) {}
        fn progress(&self) -> Status {
            self.0.get()
        }
    }

    fn state(settled: bool, failed: bool) -> Status {
        Status {
            settled,
            failed,
            blocked: failed && !settled,
        }
    }

    fn owner() -> NativeOwner {
        NativeResources::new(Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
    }

    fn child(
        owner: &NativeOwner,
        probe: Rc<Cell<Status>>,
    ) -> Recovery<NativeChildResources, FakeProbe> {
        owner.children.set(owner.children.get() + 1);
        Recovery::with_probe(
            NativeChildResources {
                owner: owner.clone(),
                _arrays: Vec::new(),
                _stream: None,
            },
            FakeProbe(probe),
        )
    }

    #[test]
    fn child_scope_cannot_hide_pending_parent_or_sibling() {
        let owner = owner();
        let parent_state = Rc::new(Cell::new(state(false, false)));
        let parent = Recovery::with_probe(owner.clone(), FakeProbe(Rc::clone(&parent_state)));
        let first = child(&owner, Rc::new(Cell::new(state(true, false))));
        let pending = Rc::new(Cell::new(state(false, false)));
        let second = child(&owner, Rc::clone(&pending));
        drop(first);
        assert!(!check_native_status(&parent).unwrap());
        parent_state.set(state(true, false));
        assert!(!check_native_status(&parent).unwrap());
        drop(second);
        assert!(!check_native_status(&parent).unwrap());
        pending.set(state(true, false));
        crate::backend::submission_recovery::wait_for_retirement(|| {
            check_native_status(&parent).unwrap()
        });
    }

    #[test]
    fn later_healthy_parent_observation_cannot_clear_child_failure() {
        let owner = owner();
        let parent = Recovery::with_probe(
            owner.clone(),
            FakeProbe(Rc::new(Cell::new(state(true, false)))),
        );
        let failed = Rc::new(Cell::new(state(false, true)));
        drop(child(&owner, Rc::clone(&failed)));
        assert!(check_native_status(&parent).is_err());
        assert!(owner.unavailable());
        assert!(!native_resources_releasable(&parent));
        failed.set(state(true, true));
        crate::backend::submission_recovery::reap();
        crate::backend::submission_recovery::wait_for_retirement(|| owner.children.get() == 0);
        assert_eq!(owner.children.get(), 0);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            native_resources_releasable(&parent)
        });
        assert!(check_native_status(&parent).is_err());
    }

    #[test]
    fn child_unwind_marks_failure_before_pending_ticket_can_drop() {
        let owner = owner();
        owner.status.set(state(true, false));
        let pending = Rc::new(Cell::new(state(false, false)));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _child = child(&owner, Rc::clone(&pending));
            let _unwind = ChildUnwind(&owner.host_failed);
            panic!("injected child observation unwind");
        }));
        assert!(result.is_err());
        assert!(owner.host_failed.get());
        assert!(owner.unavailable());
        assert_eq!(owner.children.get(), 1);
        pending.set(state(true, false));
        crate::backend::submission_recovery::reap();
        crate::backend::submission_recovery::wait_for_retirement(|| owner.children.get() == 0);
        assert_eq!(owner.children.get(), 0);
    }
}

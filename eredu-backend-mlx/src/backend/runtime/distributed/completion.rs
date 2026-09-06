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

impl Retention for NativeResources {
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
    ) -> Rc<Self> {
        let resources = Rc::new(Self {
            event: RefCell::new(None),
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
        if !resources.groups.is_empty() {
            // Reserve registration before submission, not on the failure path.
            NATIVE_RESOURCE_OWNERS.with(|owners| {
                let mut owners = owners.borrow_mut();
                owners.retain(|owner| owner.strong_count() != 0);
                owners.push(Rc::downgrade(&resources));
            });
        }
        resources
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
    owner: Rc<NativeResources>,
    _arrays: Vec<Array>,
    _stream: Option<Stream>,
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
    owner: &Rc<NativeResources>,
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
        owner: Rc::clone(owner),
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

fn check_native_status<P: Probe>(
    recovery: &Recovery<Rc<NativeResources>, P>,
) -> safemlx::error::Result<bool> {
    crate::backend::submission_recovery::reap();
    let status = recovery.progress();
    let resources = recovery.retention();
    if status.failed
        || status.blocked
        || resources.host_failed.get()
        || resources.child_failed.get()
    {
        return Err(safemlx::error::Exception::custom(
            "native communication failed; unresolved resources remain retained",
        ));
    }
    Ok(status.settled && resources.children.get() == 0)
}

mod communication;
mod generic;

thread_local! {
    static NATIVE_RESOURCE_OWNERS: RefCell<Vec<std::rc::Weak<NativeResources>>> = const { RefCell::new(Vec::new()) };
    static DISTRIBUTED_COMPLETION_ORPHANS: RefCell<generic::DistributedCompletionOrphanQuarantine> =
        RefCell::new(generic::DistributedCompletionOrphanQuarantine::default());
    static COMMUNICATION_ORPHANS: RefCell<communication::CommunicationOrphanQuarantine> =
        RefCell::new(communication::CommunicationOrphanQuarantine::default());
    #[cfg(test)]
    static FORCE_NEXT_COMMUNICATION_PENDING: Cell<bool> = const { Cell::new(false) };
}

pub(crate) use communication::ensure_group_available;
#[cfg(test)]
pub(crate) use communication::{
    distributed_completion_orphan_count, force_next_communication_pending,
    release_forced_pending_orphans,
};
pub use communication::{synchronize_outputs, MlxCommunicationCompletion, MlxFailureAgreement};
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

    fn owner() -> Rc<NativeResources> {
        NativeResources::new(Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
    }

    fn child(
        owner: &Rc<NativeResources>,
        probe: Rc<Cell<Status>>,
    ) -> Recovery<NativeChildResources, FakeProbe> {
        owner.children.set(owner.children.get() + 1);
        Recovery::with_probe(
            NativeChildResources {
                owner: Rc::clone(owner),
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
        let parent = Recovery::with_probe(Rc::clone(&owner), FakeProbe(Rc::clone(&parent_state)));
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
        assert!(check_native_status(&parent).unwrap());
    }

    #[test]
    fn later_healthy_parent_observation_cannot_clear_child_failure() {
        let owner = owner();
        let parent = Recovery::with_probe(
            Rc::clone(&owner),
            FakeProbe(Rc::new(Cell::new(state(true, false)))),
        );
        let failed = Rc::new(Cell::new(state(false, true)));
        drop(child(&owner, Rc::clone(&failed)));
        assert!(check_native_status(&parent).is_err());
        assert!(owner.unavailable());
        failed.set(state(true, true));
        crate::backend::submission_recovery::reap();
        assert_eq!(owner.children.get(), 0);
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
        assert_eq!(owner.children.get(), 0);
    }
}

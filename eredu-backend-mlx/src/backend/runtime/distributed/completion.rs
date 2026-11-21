//! Backend-independent completion ownership for distributed operations.
//!
//! Distributed MLX primitives are lazy just like ordinary array operations.
//! This module couples a submitted result with the exact `safemlx` event which
//! completes it, so pipeline and collective callers do not need to duplicate
//! `eval` plus whole-stream synchronization sequences.

use safemlx::{transforms::async_eval_with_event, Array, Event, EventBackend, Stream};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use crate::backend::{
    error::Error,
    runtime::distributed::{topology::CommunicationRouteRealization, Group},
    submission_recovery::{Recovery, Retention, Status},
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

fn native_resources_releasable(recovery: &Recovery<Rc<NativeResources>>) -> bool {
    safemlx::try_with_submission_retirement(|| {
        crate::backend::submission_recovery::reap();
        recovery.progress().settled && recovery.retention().children.get() == 0
    })
    .unwrap_or(false)
}

fn check_native_status(recovery: &Recovery<Rc<NativeResources>>) -> safemlx::error::Result<bool> {
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
}

pub(crate) use communication::ensure_group_available;
pub use communication::{synchronize_outputs, MlxCommunicationCompletion, MlxFailureAgreement};
pub use generic::DistributedCompletion;

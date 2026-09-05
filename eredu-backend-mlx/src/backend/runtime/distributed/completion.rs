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
};

mod communication;
mod generic;

thread_local! {
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

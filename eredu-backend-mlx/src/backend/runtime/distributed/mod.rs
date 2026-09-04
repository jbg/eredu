//! Architecture-independent distributed execution infrastructure.

/// Backend-independent ownership for submitted distributed work.
pub mod completion;
mod group;
/// Architecture-neutral tensor-parallel planning and execution contexts.
pub mod parallel;
/// Runtime topology, placement planning, and selective checkpoint loading.
pub mod topology;

#[cfg(all(test, unix))]
mod communication_tests;

pub(crate) use group::{
    all_gather, all_gather_for, all_gather_unchecked, all_sum, all_sum_for, all_to_all_v,
    payload_free_all_sum_for, recv, recv_like, send, Group,
};

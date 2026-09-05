//! MLX realization of neutral distributed execution plans.

/// MLX ownership for submitted distributed work.
pub mod completion;
mod group;
/// Native tensor-parallel execution contexts.
pub mod parallel;
/// Native topology realization and selective checkpoint loading.
pub mod topology;

#[cfg(all(test, unix))]
mod communication_tests;

pub(crate) use group::{
    all_gather, all_gather_for, all_gather_unchecked, all_sum, all_sum_for, all_to_all_v,
    payload_free_all_sum_for, recv, recv_like, send, Group,
};

#[cfg(test)]
pub(crate) use group::{contracted_collective_submissions, reset_native_collective_submissions};

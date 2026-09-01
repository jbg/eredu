//! Architecture-independent distributed execution infrastructure.

/// Backend-independent ownership for submitted distributed work.
pub mod completion;
mod group;
/// Architecture-neutral tensor-parallel planning and execution contexts.
pub mod parallel;
/// Runtime topology, placement planning, and selective checkpoint loading.
pub mod topology;

pub(crate) use group::{all_gather, all_sum, all_to_all_v, recv, send, Group};

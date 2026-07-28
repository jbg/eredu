//! Attention-cache storage and residency.

/// Attention key/value cache implementations.
pub mod kv;
/// Block-addressable attention-cache residency and persistence.
pub mod residency;

pub use kv::*;

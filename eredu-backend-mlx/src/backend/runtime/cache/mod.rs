//! Attention-cache storage and residency.

pub(crate) mod kv;
/// Block-addressable attention-cache residency and persistence.
pub mod residency;
/// Runtime-policy-selected key/value state realization.
pub mod state;

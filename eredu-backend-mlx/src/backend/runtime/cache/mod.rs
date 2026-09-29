//! Attention-cache storage and residency.

pub(crate) mod kv;
/// Block-addressable attention-cache residency and persistence.
pub mod residency;
/// Runtime-policy-selected key/value state realization.
pub mod state;

/// Named bounded append-stream storage and transactional checkpoint.
pub use kv::MlxPagedAppendStream;

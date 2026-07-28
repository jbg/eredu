//! Architecture-independent model execution infrastructure.

/// Attention-cache storage and residency.
pub mod cache;
/// Checkpoint loading, storage, binding, and conversion.
pub mod checkpoint;
/// Generic execution engines and observability hooks.
pub mod execution;
/// Weight-residency planning and execution.
pub mod residency;

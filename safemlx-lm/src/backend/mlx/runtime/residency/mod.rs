//! Weight-residency planning and execution.

/// Experimental bounded streaming of dense execution units.
pub mod dense_stream;
/// Architecture-independent sparse routed-expert caching.
pub mod expert_cache;
/// Budgeted residency manager for logical immutable weight units.
pub mod manager;

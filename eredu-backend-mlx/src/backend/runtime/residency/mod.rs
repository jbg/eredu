//! Weight-residency planning and execution.

/// Experimental bounded streaming of dense execution units.
pub mod dense_stream;
/// Budgeted residency manager for logical immutable weight units.
pub mod manager;
/// Architecture-independent addressable parameter-bank residency.
pub mod parameter_bank;

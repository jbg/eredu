//! This mod defines the traits for neural network modules and parameters.
//!
//! Keeping these traits separate from the neural-network modules also allows downstream crates to
//! use the `safemlx_macros::ModuleParameters` derive macro for their own module implementations.

#[allow(clippy::module_inception)]
mod module;
mod param;

pub use module::*;
pub use param::*;

//! Traits for backend neural-network modules and parameters.
//!
//! Keeping these traits separate from the neural-network modules also allows downstream crates to
//! use the `eredu_backend_mlx_macros::ModuleParameters` derive macro for their own module implementations.

#[allow(clippy::module_inception)]
mod module;
mod param;

pub use module::*;
pub use param::*;

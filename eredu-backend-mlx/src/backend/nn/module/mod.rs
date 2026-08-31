//! Private physical-slot traversal for native MLX kernels.
//!
//! Architecture and composition code use `eredu_nn::Parameterized`; this
//! module must not be exported as a second parameter API.

#[allow(clippy::module_inception)]
mod module;
mod param;

pub use module::*;
pub use param::*;

//! MLX implementations of Eredu's backend-neutral contracts.
//!
//! This crate is the concrete MLX boundary. It may depend on neutral Eredu
//! crates and SafeMLX, but it must not depend on the `eredu` facade crate.

#![allow(missing_docs)]
#![allow(
    clippy::arc_with_non_send_sync,
    clippy::drop_non_drop,
    clippy::large_enum_variant,
    clippy::new_ret_no_self,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

/// Concrete MLX backend implementation and runtime infrastructure.
pub mod backend;
/// Optional MLX bindings for backend-neutral audio codecs.
#[cfg(feature = "codec")]
pub mod codec;
/// MLX bindings for Eredu's backend-neutral model-family definitions.
pub mod composition;

/// Backend-neutral core contracts used by the extracted implementation.
pub use eredu_core as core;

/// Deliberate access to the native MLX handles needed by applications that
/// configure devices, streams, memory, or low-level tensor inputs.
pub mod native {
    pub use safemlx::*;
}

mod tensor;

#[cfg(test)]
mod test_utils;

pub use tensor::MlxTensor;

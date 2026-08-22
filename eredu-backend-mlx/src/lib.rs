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

mod adapter;
#[allow(dead_code, unused_imports)]
mod backend;
/// Optional MLX bindings for backend-neutral audio codecs.
#[cfg(feature = "codec")]
pub mod codec;
#[allow(dead_code, unused_imports)]
mod composition;

pub use adapter::*;

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

/// Internal fixtures shared with Eredu's backend integration-test package.
///
/// This module is not part of the production adapter API. It is available only
/// when the explicit `test-support` feature is enabled.
#[cfg(feature = "test-support")]
#[doc(hidden)]
pub mod testing {
    pub mod backend {
        pub use crate::backend::mlx;
    }

    pub mod composition {
        pub use crate::composition::*;
    }
}

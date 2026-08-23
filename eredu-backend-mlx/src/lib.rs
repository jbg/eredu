#![doc = include_str!("../README.md")]
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
/// Reusable MLX tensors, operators, runtime facilities, and distributed primitives.
pub mod backend;
/// Optional MLX bindings for backend-neutral audio codecs.
#[cfg(feature = "codec")]
pub mod codec;
mod composition;

pub use adapter::*;

/// Backend-neutral core contracts used by the extracted implementation.
pub use eredu_core as core;

/// Deliberate access to the native MLX handles needed by applications that
/// configure devices, streams, memory, or low-level tensor inputs.
pub mod native {
    pub use safemlx::*;

    /// Constructs an MLX backend from explicitly selected native streams.
    pub fn backend(stream: &Stream, weights_stream: &Stream) -> crate::MlxBackend<'static> {
        crate::MlxBackend::new(stream, weights_stream)
    }

    /// Constructs a distributed MLX backend from native streams and a world group.
    pub fn distributed_backend<'a>(
        stream: &Stream,
        weights_stream: &Stream,
        world: &'a distributed::Group,
    ) -> crate::MlxBackend<'a> {
        crate::MlxBackend::with_distributed_world(stream, weights_stream, world)
    }
}

mod tensor;

#[cfg(test)]
mod test_utils;

pub use tensor::MlxTensor;

/// Internal fixtures shared with Eredu's backend integration-test package.
///
/// This module is not part of the production adapter API. It is available only
/// when the explicit `test-support` feature is enabled.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub mod testing {
    pub mod backend {
        pub use crate::backend::mlx;
    }

    pub mod composition {
        pub use crate::composition::*;
    }
}

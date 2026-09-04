//! Safe, low-level Rust bindings to MLX.
//!
//! `safemlx` exposes MLX arrays, devices, streams, native operators, function
//! transforms, and accelerator/runtime facilities. It intentionally does not
//! define neural-network layers, models, optimizers, checkpoint policy, or
//! other framework abstractions; those belong to consumers such as
//! `eredu-backend-mlx`.
//!
//! Operations are lazy and are scheduled on an explicit [`Stream`]. Call
//! [`Array::evaluated`] (or a function in [`transforms`]) to submit work before
//! accessing host data. Native MLX `.npy` and `safetensors` I/O is exposed on
//! [`Array`]; higher-level checkpoint formats are outside this crate.

#![deny(unused_unsafe, missing_debug_implementations, missing_docs)]
#![cfg_attr(test, allow(clippy::approx_constant))]

#[macro_use]
pub mod macros;

mod array;
#[cfg(feature = "cuda")]
pub mod cuda;
mod device;
pub mod distributed;
mod dtype;
pub mod error;
mod event;
pub mod fast;
pub mod fft;
mod host_transfer;
pub mod linalg;
pub mod memory;
#[cfg(feature = "metal")]
pub mod metal;
pub mod ops;
pub mod random;
mod runtime_deadline;
mod stream;
pub mod system;
pub mod transforms;
pub mod utils;

pub use array::*;
pub use device::*;
pub use dtype::*;
pub use event::*;
pub use host_transfer::*;
pub use runtime_deadline::*;
pub use stream::*;

/// Registers one idempotent same-thread housekeeping callback.
///
/// The callback runs while the process-wide MLX runtime lock is held whenever
/// this thread next enters the runtime. Reentrant runtime calls do not invoke
/// it recursively. This lets thread-affine completion owners release finished
/// resources without a foreign polling thread.
pub fn register_thread_runtime_housekeeping(hook: fn()) {
    utils::runtime_lock::register_housekeeping_hook(hook);
}

/// Removes a previously registered same-thread housekeeping callback.
pub fn unregister_thread_runtime_housekeeping(hook: fn()) {
    utils::runtime_lock::unregister_housekeeping_hook(hook);
}

#[cfg(test)]
pub(crate) fn test_stream() -> &'static Stream {
    Box::leak(Box::new(Stream::new_with_device(&Device::new(
        DeviceType::Cpu,
        0,
    ))))
}

#[cfg(test)]
pub(crate) fn test_key(seed: u64, stream: &Stream) -> Array {
    use crate::ops::indexing::TryIndexOp;

    random::split_n(random::key(seed).unwrap(), 2, stream)
        .unwrap()
        .try_index_device(1, stream)
        .unwrap()
}

#[cfg(test)]
pub(crate) fn test_concurrency() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(2)
        .clamp(2, 16)
}

pub(crate) mod constants {
    pub(crate) const DEFAULT_STACK_VEC_LEN: usize = 4;
}

pub(crate) mod sealed {
    pub trait Sealed {}

    impl Sealed for () {}
    impl<A> Sealed for (A,) where A: Sealed {}
    impl<A, B> Sealed for (A, B)
    where
        A: Sealed,
        B: Sealed,
    {
    }
    impl<A, B, C> Sealed for (A, B, C)
    where
        A: Sealed,
        B: Sealed,
        C: Sealed,
    {
    }
}

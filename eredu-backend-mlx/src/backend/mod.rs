//! MLX backend adapter.

pub(crate) mod compaction;
/// Session-owned MLX communicators, transfers, and collectives.
pub mod distributed;
/// Errors produced by MLX model loading and execution.
pub mod error;
mod execution;
#[cfg(any(feature = "image", feature = "audio"))]
mod media;
/// Reusable MLX neural-network building blocks.
pub mod nn;
pub(crate) mod ordinary_retirement;
/// Stateful random-key ownership for backend sessions.
pub mod random;
/// MLX allocator observations for neutral residency telemetry.
pub mod residency;
/// MLX-only tensor, checkpoint, execution, and residency infrastructure.
pub mod runtime;
pub(crate) mod submission_recovery;
/// MLX process-local device binding for composition-owned rank topology.
pub(crate) mod topology;
pub(crate) use distributed::MlxDistributedSession;
pub use execution::ExecutionContext;
pub use topology::{DeviceAssignment, MlxRankContext};

use eredu_core::backend::{
    BackendDescriptor, BackendProvider, Completion, DeviceCapabilities, DeviceDescriptor,
    ModelLoadingBackend, PreparedModel, SessionCapabilities, Submission,
};
use std::num::NonZeroU8;

use safemlx::{transforms::async_eval_with_event, Array, Device, DeviceType, Event, Stream};

#[cfg(any(feature = "image", feature = "audio"))]
use crate::composition::mlx::ModelProcessor;
use crate::{
    backend::error::Error,
    composition::mlx::{Executable, MlxModelSession},
    MlxLoadRequest,
};

mod adapter;

pub use adapter::completion::MlxCompletion;
#[cfg(test)]
use adapter::device::device_capabilities;
pub(crate) use adapter::device::{MlxAcceleratorFamily, MlxDeviceIdentity};
pub use adapter::model::MlxModel;
pub use adapter::provider::MlxBackend;
pub(crate) use adapter::target::{validate_native_execution_target, MlxPreparedTarget};

#[cfg(test)]
#[path = "adapter/tests.rs"]
mod tests;

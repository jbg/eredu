#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
#![allow(
    clippy::arc_with_non_send_sync,
    clippy::drop_non_drop,
    clippy::large_enum_variant,
    clippy::new_ret_no_self,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

#[cfg(all(feature = "metal", feature = "cuda"))]
compile_error!("the `metal` and `cuda` backend features are mutually exclusive");

mod adapter;
/// Reusable MLX tensors, operators, runtime facilities, and distributed primitives.
pub mod backend;
mod composition;

pub use adapter::*;

pub(crate) use backend::nn::{module, native_quantization, nested, primitives as nn};
pub(crate) use safemlx::{ops, Array, Dtype, Stream};

/// Composition-owned MLX integrations for backend tooling.
///
/// Native arrays, devices, streams, operations, and collectives retain their
/// canonical [`safemlx`] paths and are not re-exported here. Reusable MLX
/// backend facilities are organized under [`backend`].
pub mod native {
    pub use crate::backend::runtime::distributed::topology::NativeWorldManifestTransport;
    #[cfg(any(feature = "image", feature = "audio"))]
    pub use crate::composition::mlx::MlxMediaInputLowering;

    pub use crate::composition::mlx::component_binding::MlxComponentBinder;
    use safemlx::Stream;

    pub use crate::backend::distributed::{MlxRealtimeConsensusTransport, MlxRingTransport};
    pub use crate::backend::runtime::checkpoint::quantization::MlxSafetensorsWeightPacker;
    pub use crate::backend::topology::DeviceAssignment;
    pub use crate::backend::{random::RandomState, ExecutionContext};
    pub use crate::composition::mlx::realtime::{
        MlxRealtimeBinding, MlxRealtimeCompletion, MlxRealtimeExecutionContext, MlxRealtimeQueue,
        MlxRealtimeSupport,
    };
    pub use crate::composition::mlx::speculative::MlxAssistantBinder;
    pub use crate::composition::mlx::{
        inspection_media_features, MlxModelOutput, MlxSessionCompletion, MlxSessionMechanisms,
    };

    /// Constructs an MLX backend from explicitly selected native streams.
    pub fn backend(
        stream: &Stream,
        weights_stream: &Stream,
    ) -> crate::backend::MlxBackend<'static> {
        crate::backend::MlxBackend::new(stream, weights_stream)
    }

    /// Constructs a distributed MLX backend from native streams and a world group.
    pub fn distributed_backend<'a>(
        stream: &Stream,
        weights_stream: &Stream,
        world: &'a safemlx::distributed::Group,
    ) -> crate::backend::MlxBackend<'a> {
        crate::backend::MlxBackend::with_distributed_world(stream, weights_stream, world)
    }
}

mod tensor;

pub use tensor::MlxTensor;

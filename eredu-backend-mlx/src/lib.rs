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

#[cfg(all(feature = "metal", feature = "cuda"))]
compile_error!("the `metal` and `cuda` backend features are mutually exclusive");

mod adapter;
/// Reusable MLX tensors, operators, runtime facilities, and distributed primitives.
pub mod backend;
/// Optional MLX bindings for backend-neutral audio codecs.
#[cfg(feature = "codec")]
pub mod codec;
mod composition;

pub use adapter::*;

/// Deliberate access to native MLX handles for backend-author tools that
/// configure devices, streams, memory, or low-level tensor inputs.
///
/// Native session, completion, drafting, input, and realtime types are not
/// part of the flat curated adapter:
///
/// ```compile_fail
/// use eredu_backend_mlx::MlxCompletion;
/// ```
///
/// ```compile_fail
/// use eredu_backend_mlx::MlxDrafter;
/// ```
///
/// ```compile_fail
/// use eredu_backend_mlx::MlxModelInput;
/// ```
///
/// ```compile_fail
/// use eredu_backend_mlx::MlxModelSession;
/// ```
///
/// ```compile_fail
/// use eredu_backend_mlx::MlxRealtimeInput;
/// ```
///
/// ```compile_fail
/// use eredu_backend_mlx::MlxSessionCompletion;
/// ```
///
/// Prepared models, native-bearing load and inspection policy, backend errors,
/// model-session outputs, and checkpoint conversion also require an explicit
/// native import:
///
/// ```compile_fail
/// use eredu_backend_mlx::MlxModel;
/// ```
///
/// ```compile_fail
/// use eredu_backend_mlx::MlxModelConfig;
/// ```
///
/// ```compile_fail
/// use eredu_backend_mlx::ModelLoadOptions;
/// ```
///
/// ```compile_fail
/// use eredu_backend_mlx::MlxError;
/// ```
///
/// ```compile_fail
/// use eredu_backend_mlx::MlxInspectionOptions;
/// ```
///
/// ```compile_fail
/// use eredu_backend_mlx::inspect_model;
/// ```
///
/// ```compile_fail
/// use eredu_backend_mlx::MlxModelOutput;
/// ```
///
/// ```compile_fail
/// use eredu_backend_mlx::quantize_checkpoint;
/// ```
///
/// ```compile_fail
/// use eredu_backend_mlx::CheckpointQuantizationOptions;
/// ```
///
/// Device-bound topology also stays behind the backend-author boundary:
///
/// ```compile_fail
/// use eredu_backend_mlx::DeviceAssignment;
/// ```
///
/// ```compile_fail
/// use eredu_backend_mlx::MlxParallelContext;
/// ```
///
/// Raw completion submission is internal even though the opaque completion
/// type also participates in the public reusable backend implementation:
///
/// ```compile_fail
/// use eredu_backend_mlx::{backend::MlxCompletion, native::Array};
///
/// fn submit(output: Array) {
///     let _ = MlxCompletion::submission(output);
/// }
/// ```
pub mod native {
    pub use crate::backend::nn::generation::sample;
    pub use crate::backend::runtime::checkpoint::quantization::{
        CheckpointQuantizationOptions, CheckpointQuantizationReport,
    };
    pub use crate::backend::runtime::generation::sampler::Sampler;
    pub use crate::backend::{
        error::Error as MlxError, DeviceAssignment, MlxCompletion, MlxModel, MlxModelConfig,
        MlxParallelContext, ModelLoadOptions,
    };
    pub use crate::composition::mlx::realtime::personaplex_prompt::sine_frame as personaplex_sine_frame;
    pub use crate::composition::mlx::realtime::{
        MlxRealtimeBackend, MlxRealtimeCompletion, MlxRealtimeInput, MlxRealtimeModel,
        MlxRealtimeModelIdentity, MlxRealtimeModelState, MlxRealtimeModelStateBranch,
        MlxRealtimeOutput, MlxRealtimeSession, MlxRealtimeSessionBranch,
    };
    pub use crate::composition::mlx::speculative::MlxDrafter;
    pub use crate::composition::mlx::{
        inspect_model, MlxInspectionOptions, MlxModelInput, MlxModelOutput, MlxModelSession,
        MlxSessionCompletion,
    };
    pub use safemlx::*;

    /// Converts a checkpoint with an explicitly selected native execution stream.
    pub fn quantize_checkpoint(
        source_dir: impl AsRef<std::path::Path>,
        output_dir: impl AsRef<std::path::Path>,
        options: &CheckpointQuantizationOptions,
        stream: &Stream,
    ) -> Result<CheckpointQuantizationReport, MlxError> {
        crate::backend::runtime::checkpoint::quantization::quantize_checkpoint(
            source_dir, output_dir, options, stream,
        )
    }

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

#[cfg(test)]
mod tests;

pub use tensor::MlxTensor;

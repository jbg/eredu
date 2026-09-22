#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
// Retained error auto-trait proofs traverse source accounts, prepared pin groups
// and the deliberate pool -> quarantined source pin -> account -> pool cycle.
// Keep those owners intact; this affects compiler proof depth, not runtime stack
// size, allocation limits, original custody, or the workspace unsafe-code policy.
#![recursion_limit = "256"]
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
pub use backend::managed_memory::{configure_memory_limits, memory_snapshot, memory_topology};
pub use composition::mlx::{MlxLoadRequest, MlxModelConfig, MlxSelectedPreparation};

pub(crate) use backend::nn::{module, native_quantization, nested, primitives as nn};
pub(crate) use safemlx::{ops, Array, Dtype, Stream};

#[cfg(test)]
pub(crate) use safemlx::{array, Device};

#[cfg(test)]
pub(crate) fn test_stream() -> &'static Stream {
    Box::leak(Box::new(Stream::new_with_device(&safemlx::Device::new(
        safemlx::DeviceType::Cpu,
        0,
    ))))
}

#[cfg(test)]
pub(crate) mod array {
    use safemlx::{Array, ArrayElement};

    pub(crate) fn eval_vec<T>(array: &Array) -> Vec<T>
    where
        T: ArrayElement + Clone,
    {
        array.evaluated().unwrap().as_slice::<T>().to_vec()
    }

    pub(crate) fn eval_equal_values(lhs: &Array, rhs: &Array) -> bool {
        let lhs = lhs.evaluated().unwrap();
        let rhs = rhs.evaluated().unwrap();
        lhs.equal_values(&rhs)
    }
}

/// Composition-owned MLX integrations for backend tooling.
///
/// Native arrays, devices, streams, operations, and collectives retain their
/// canonical [`safemlx`] paths and are not re-exported here. Reusable MLX
/// backend facilities are organized under [`backend`].
pub mod native {
    pub use crate::composition::moshi::MlxRealtimeExecution;
    use safemlx::Stream;

    pub use crate::backend::distributed::MlxRealtimeConsensusTransport;
    pub use crate::backend::topology::DeviceAssignment;
    pub use crate::backend::{random::RandomState, ExecutionContext};
    pub use crate::composition::mlx::realtime::{
        MlxManagedFrameSessionBranch, MlxManagedRealtimeScheduler, MlxManagedRealtimeSessionState,
        MlxPreparedRealtimeExecution, MlxRealtimeCompletion, MlxRealtimeExecutionContext,
        MlxRealtimeFrameCompletionMechanism, MlxRealtimeFramePreparation,
        MlxRealtimeFrameTensorMechanisms, MlxRealtimeHostObserver,
    };
    pub use crate::composition::mlx::speculative::MlxDrafter;
    pub use crate::composition::mlx::{
        inspect_model, inspect_model_preparation, MlxHostInputUploadError, MlxInspectionOptions,
        MlxModelInput, MlxModelOutput, MlxModelSession, MlxNativeTextState,
        MlxOriginalPreparedModelInput, MlxOriginalPreparedNativeInput,
        MlxPreparedInputMaterializer, MlxPreparedModelInputBindError, MlxPreparedModelInputError,
        MlxPreparedModelInputPlan, MlxPreparedNativeInputError, MlxPreparedNativeInputPlan,
        MlxSessionCompletion, MlxTextSamplingState,
    };
    /// Converts a checkpoint with an explicitly selected native execution stream.
    pub fn quantize_checkpoint(
        source_dir: impl AsRef<std::path::Path>,
        output_dir: impl AsRef<std::path::Path>,
        options: &crate::backend::runtime::checkpoint::quantization::CheckpointQuantizationOptions,
        stream: &Stream,
    ) -> Result<
        crate::backend::runtime::checkpoint::quantization::CheckpointQuantizationReport,
        crate::backend::error::Error,
    > {
        crate::backend::runtime::checkpoint::quantization::quantize_checkpoint(
            source_dir, output_dir, options, stream,
        )
    }

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

    /// Native prepared stream failure retaining its actual incomplete owner.
    pub use crate::backend::managed_memory::gpu_stream::MlxGpuStreamError;

    /// Constructs a distributed backend retaining the same admitted execution
    /// and source stream owners as the execution-plan factory.
    ///
    /// The supplied world remains the actual communicator. This does not adopt
    /// ordinary stream aliases or grant native storage from a caller capacity.
    /// `Ok(None)` preserves the factory's explicit absence of a qualified stream
    /// constructor. The native error retains its original incomplete owners.
    pub fn prepared_distributed_backend(
        world: &safemlx::distributed::Group,
    ) -> Result<Option<crate::backend::MlxBackend<'_>>, MlxGpuStreamError> {
        prepared_distributed_backend_on(world, safemlx::DeviceType::Gpu)
    }

    /// Constructs prepared distributed streams on the selected native device
    /// type at index zero, using the same mechanism choice as the plan factory.
    /// The world and exact native stream owners remain retained by the backend.
    pub fn prepared_distributed_backend_on(
        world: &safemlx::distributed::Group,
        device: safemlx::DeviceType,
    ) -> Result<Option<crate::backend::MlxBackend<'_>>, MlxGpuStreamError> {
        let streams = crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams::for_device_factory(
            &crate::backend::managed_memory::ledger(), device,
        )?;
        Ok(streams.map(|streams| {
            crate::backend::MlxBackend::with_prepared_distributed_world(streams, world)
        }))
    }

    /// Binds semantic topology, wire dtype, and maximum invocation geometry to model options.
    ///
    /// The batch and sequence limits resolve architecture-owned boundary
    /// schemas during selection and must cover every later invocation.
    pub fn parallel_load_options(
        topology: eredu_core::ParallelRankTopology,
        device: DeviceAssignment,
        pipeline_wire: eredu_runtime::PipelineWireContract,
        maximum_batch_size: i32,
        maximum_sequence_length: i32,
        completion_policy: eredu_runtime::CommunicationCompletionPolicy,
    ) -> Result<crate::MlxLoadRequest, crate::backend::error::Error> {
        crate::MlxLoadRequest::with_parallel(
            topology,
            device,
            pipeline_wire,
            maximum_batch_size,
            maximum_sequence_length,
            completion_policy,
        )
    }
}

mod tensor;

#[cfg(test)]
pub(crate) fn test_parallel_rank(
    rank: usize,
    tensor: usize,
    pipeline: usize,
    expert: usize,
) -> eredu_core::ParallelRankTopology {
    eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(tensor, pipeline, expert, 1)
            .expect("test topology is valid"),
        rank,
    )
    .expect("test rank belongs to topology")
}

#[cfg(test)]
mod tests;

pub use tensor::MlxTensor;

#[cfg(test)]
pub(crate) mod memory_fixture;

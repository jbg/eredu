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
    pub use crate::backend::runtime::distributed::coordinator::MlxReplicatedCoordinator;
    /// Reports exact cold cache mechanisms for an architecture-declared state layout.
    /// This allocates no native device, stream, or cache storage.
    pub fn state_capabilities(
        requirements: &eredu_runtime::StateRealizationRequirements,
        policy: &eredu_runtime::CacheResidencyPolicy,
    ) -> eredu_runtime::StateMechanismCapabilities {
        crate::composition::mlx::state_capabilities(requirements, policy)
    }

    pub use crate::composition::moshi::MlxRealtimeExecution;
    use safemlx::Stream;

    pub use crate::backend::distributed::MlxRealtimeConsensusTransport;
    pub use crate::backend::topology::DeviceAssignment;
    pub use crate::backend::{random::RandomState, ExecutionContext};
    pub use crate::composition::mlx::realtime::{
        MlxPreparedRealtimeExecution, MlxRealtimeCompletion, MlxRealtimeCompletionMechanism,
        MlxRealtimeExecutionContext, MlxRealtimeFrameTensorMechanisms, MlxRealtimeHostObserver,
    };
    pub use crate::composition::mlx::speculative::MlxDrafter;
    pub use crate::composition::mlx::voicechat::{
        initialize_voicechat_neural_state, materialize_ear_decoder,
        materialize_ear_decoder_parallel, materialize_ear_decoder_pipeline,
        materialize_ear_encoder, materialize_ear_encoder_parallel,
        materialize_ear_encoder_pipeline, materialize_prvq, materialize_prvq_parallel,
        materialize_voicechat_backbone, materialize_voicechat_components,
        materialize_voicechat_conditioning, materialize_voicechat_conditioning_parallel,
        materialize_voicechat_function_head, materialize_voicechat_function_head_parallel,
        materialize_voicechat_mog, materialize_voicechat_mog_parallel,
        materialize_voicechat_perception, materialize_voicechat_perception_parallel,
        materialize_voicechat_perception_pipeline, materialize_voicechat_perception_rnnt_pipeline,
        materialize_voicechat_rnnt, materialize_voicechat_rnnt_parallel,
        materialize_voicechat_rnnt_predictor_pipeline, materialize_voicechat_rnnt_projection,
        materialize_voicechat_rnnt_projection_parallel, materialize_voicechat_rvq,
        materialize_voicechat_rvq_parallel, materialize_voicechat_speech_providers,
        materialize_voicechat_tts_transformer, materialize_voicechat_tts_transformer_parallel,
        materialize_voicechat_tts_transformer_partition,
        materialize_voicechat_tts_transformer_pipeline, MaterializedMlxSpeechProviders,
        MlxConditioning, MlxEarDecoder, MlxEarDecoderPipeline, MlxEarDecoderPipelineState,
        MlxEarEncoder, MlxEarEncoderPipeline, MlxFunctionHead, MlxMog, MlxPerception,
        MlxPerceptionPipeline, MlxPerceptionPipelineState, MlxPerceptionRnntMetadata,
        MlxPerceptionRnntPipeline, MlxPerceptionRnntResidencyReports, MlxPrvq, MlxRnnt,
        MlxRnntPredictorPipeline, MlxRnntProjection, MlxRvq, MlxSpeechProviders, MlxTtsTransformer,
        MlxTtsTransformerPartition, MlxTtsTransformerPipeline, MlxTtsTransformerPipelineState,
        MlxVoiceChatBackbone, MlxVoiceChatBranch, MlxVoiceChatComponents, MlxVoiceChatExecution,
        MlxVoiceChatGeneration, MlxVoiceChatPrepublication, VoiceChatNeuralInitializationError,
        VoiceChatSubmissionError,
    };
    pub use crate::composition::mlx::{
        inspect_model, inspect_model_preparation, MlxInspectionOptions, MlxModelInput,
        MlxModelOutput, MlxModelSession, MlxNativeTextState, MlxSessionCompletion,
        MlxTextSamplingState,
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

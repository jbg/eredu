//! Binding of the architecture's joint neural transaction to native cache storage.

use super::replicated_text::MlxStateMechanisms;
use crate::backend::runtime::cache::state::{
    JointCopyLimits, MlxHybridState, MlxJointState, MlxJointStateBranch, MlxKeyValueLayerState,
    MlxKeyValueState,
};
use eredu_architectures::voicechat::state::{
    NeuralStateGeometry, NeuralStateGeometryError, NeuralStateParts,
};
use eredu_runtime::SelectedStateRealization;

use crate::{
    backend::{nn::shared::MlxNeuralBackend, submission_recovery::detached_preparation},
    MlxTensor,
};
use eredu_architectures::voicechat::{
    component_residency::VoiceChatResidencyPolicy,
    components::{PreparedVoiceChatConstruction, VoiceChatComponents, VoiceChatConstructionError},
    frontend::FrontendOptions,
    input::InputState,
};
use safemlx::{Array, Stream};

mod rnnt;
pub use rnnt::{materialize_voicechat_rnnt, materialize_voicechat_rnnt_parallel, MlxRnnt};
mod rnnt_projection;
pub use rnnt_projection::{
    materialize_voicechat_rnnt_projection, materialize_voicechat_rnnt_projection_parallel,
    MlxRnntProjection,
};
mod rnnt_predictor_pipeline;
pub use rnnt_predictor_pipeline::{
    materialize_voicechat_rnnt_predictor_pipeline, MlxRnntPredictorPipeline,
};

mod backbone;
pub use backbone::{materialize_voicechat_backbone, MlxVoiceChatBackbone};

mod codec_quantizer;
pub use codec_quantizer::{materialize_prvq, materialize_prvq_parallel, MlxPrvq};

mod codec_encoder;
pub use codec_encoder::{materialize_ear_encoder, materialize_ear_encoder_parallel, MlxEarEncoder};
mod codec_encoder_pipeline;
pub use codec_encoder_pipeline::{materialize_ear_encoder_pipeline, MlxEarEncoderPipeline};
mod codec_decoder;
pub use codec_decoder::{materialize_ear_decoder, materialize_ear_decoder_parallel, MlxEarDecoder};

mod codec_decoder_pipeline;
pub use codec_decoder_pipeline::{
    materialize_ear_decoder_pipeline, MlxEarDecoderPipeline, MlxEarDecoderPipelineState,
};

mod providers;
pub use providers::{
    materialize_voicechat_speech_providers, MaterializedMlxSpeechProviders, MlxSpeechProviders,
};

mod function_head;
pub use function_head::{
    materialize_voicechat_function_head, materialize_voicechat_function_head_parallel,
    MlxFunctionHead,
};

mod conditioning;
pub use conditioning::{
    materialize_voicechat_conditioning, materialize_voicechat_conditioning_parallel,
    MlxConditioning,
};

mod rvq;
pub use rvq::{materialize_voicechat_rvq, materialize_voicechat_rvq_parallel, MlxRvq};

mod mog;
pub use mog::{materialize_voicechat_mog, materialize_voicechat_mog_parallel, MlxMog};

mod tts_transformer_partition;
pub use tts_transformer_partition::{
    materialize_voicechat_tts_transformer_partition, MlxTtsTransformerPartition,
};

mod tts_transformer_pipeline;
pub use tts_transformer_pipeline::{
    materialize_voicechat_tts_transformer_pipeline, MlxTtsTransformerPipeline,
    MlxTtsTransformerPipelineState,
};

mod tts_transformer;
pub use tts_transformer::{
    materialize_voicechat_tts_transformer, materialize_voicechat_tts_transformer_parallel,
    MlxTtsTransformer,
};

mod perception;
pub use perception::{
    materialize_voicechat_perception, materialize_voicechat_perception_parallel, MlxPerception,
};

mod perception_pipeline;
pub use perception_pipeline::{
    materialize_voicechat_perception_pipeline, MlxPerceptionPipeline, MlxPerceptionPipelineState,
};

mod submission;
pub use submission::{
    MlxVoiceChatBranch, MlxVoiceChatExecution, MlxVoiceChatPrepublication, VoiceChatSubmissionError,
};

/// Bound VoiceChat components with independently selected speech providers.
/// The Nemotron block provider is included in total parameter admission.
pub type MlxVoiceChatComponents = VoiceChatComponents<
    MlxNeuralBackend,
    MlxHybridState,
    MlxVoiceChatBackbone,
    MlxPerception,
    MlxTtsTransformer,
    MlxMog,
    MlxRvq,
    MlxConditioning,
    MlxEarDecoder,
    MlxPrvq,
    MlxEarEncoder,
    MlxFunctionHead,
    MlxRnnt,
>;

/// The ordinary shared generation owner with coupled MLX state and retained RNG.
/// Construct through the architecture's prepared generation policy after cache
/// initialization. Submission still requires the enclosing completion owner.
pub type MlxVoiceChatGeneration = eredu_architectures::voicechat::preparation::Generation<
    MlxTensor,
    MlxJointState,
    crate::backend::random::RandomState,
    super::realtime::MlxRealtimeCompletion,
>;

/// Invalid retained cache selection or failed native initialization.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatNeuralInitializationError {
    /// Selected state differs from architecture-declared geometry.
    #[error(transparent)]
    Geometry(#[from] NeuralStateGeometryError),
    /// Native initialization failure with its original source retained.
    #[error("VoiceChat neural state initialization failed: {0}")]
    Backend(#[source] eredu_core::BackendFailure),
}

/// Creates one joint owner from exact, already selected cache realizations.
///
/// Both selections are validated before either owner is allocated. The ordinary
/// cache realization driver consumes their device/paging policy and finite pools.
/// Geometry remains architecture-owned and the shared selector guarantees
/// checkpoint/rollback support. This operation
/// neither selects capabilities nor starts generation or prompt prefill.
pub fn initialize_voicechat_neural_state(
    geometry: &NeuralStateGeometry,
    backbone: &SelectedStateRealization,
    tts: &SelectedStateRealization,
    copy_limits: JointCopyLimits,
    stream: &Stream,
) -> Result<MlxJointState, VoiceChatNeuralInitializationError> {
    geometry.validate_layouts(backbone.layout(), tts.layout())?;
    let map_error = |error| {
        VoiceChatNeuralInitializationError::Backend(eredu_core::BackendFailure::from_error(error))
    };
    let backbone = MlxHybridState::realize(backbone, None, 0).map_err(map_error)?;
    let tts = MlxKeyValueState::realize(tts, None, 0).map_err(map_error)?;
    MlxJointState::new(backbone, tts, stream, copy_limits).map_err(|error| {
        VoiceChatNeuralInitializationError::Backend(eredu_core::BackendFailure::from_error(error))
    })
}

/// Realizes an architecture-prepared component set using ordinary MLX mechanisms.
///
/// Artifact interpretation and learned-module composition remain in the portable
/// driver. This low-level entry point creates no session or neural cache owner.
/// Native materialization errors preserve their source in the portable error.
pub fn materialize_voicechat_components(
    prepared: PreparedVoiceChatConstruction<'_>,
    options: FrontendOptions,
    residency: &VoiceChatResidencyPolicy,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxVoiceChatComponents, InputState<MlxTensor>), VoiceChatConstructionError> {
    detached_preparation(
        Vec::<Array>::new(),
        || {
            let mut binder = providers::Binder::new(weights_stream, execution_stream);
            prepared.materialize::<MlxNeuralBackend, MlxHybridState, _>(
                options,
                residency,
                &mut binder,
                execution_stream,
            )
        },
        |error| VoiceChatConstructionError::Backend(eredu_core::BackendFailure::from_error(error)),
    )
}

impl NeuralStateParts for MlxJointStateBranch {
    type Backbone = MlxHybridState;
    type TtsState = [MlxKeyValueLayerState];

    fn parts_mut(&mut self) -> (&mut Self::Backbone, &mut Self::TtsState) {
        MlxJointStateBranch::parts_mut(self)
    }
}

mod perception_rnnt_pipeline;
pub use perception_rnnt_pipeline::{
    materialize_voicechat_perception_rnnt_pipeline, MlxPerceptionRnntMetadata,
    MlxPerceptionRnntPipeline, MlxPerceptionRnntResidencyReports,
};

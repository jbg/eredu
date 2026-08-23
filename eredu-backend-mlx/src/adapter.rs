//! Curated application-facing MLX adapter API.

pub use crate::backend::error::Error as MlxError;
pub use crate::backend::nn::generation::sample;
pub use crate::backend::runtime::checkpoint::quantization::{
    quantize_checkpoint, CheckpointQuantizationOptions, CheckpointQuantizationReport,
};
pub use crate::backend::runtime::generation::sampler::Sampler;
pub use crate::backend::runtime::media::input::{InputPart, ModelInput};
pub use crate::backend::runtime::residency::expert_cache::{
    ExpertCacheReport, ExpertPassStatistics, ExpertTierStatistics,
};
pub use crate::backend::{
    DeviceAssignment, MlxBackend, MlxCompletion, MlxModel, MlxModelConfig, MlxParallelContext,
    ModelLoadOptions,
};
pub use crate::composition::mlx::automatic::{
    discover_hardware, expert_cache_telemetry, mtp_telemetry, residency_telemetry,
    MlxBackendFactory,
};
pub use crate::composition::mlx::realtime::personaplex_prompt::sine_frame as personaplex_sine_frame;
pub use crate::composition::mlx::realtime::{
    generate_encoded_greedy, MlxEncodedAudioOutput, MlxRealtimeBackend, MlxRealtimeCompletion,
    MlxRealtimeInput, MlxRealtimeModel, MlxRealtimeModelIdentity, MlxRealtimeModelState,
    MlxRealtimeModelStateBranch, MlxRealtimeOutput, MlxRealtimeSession, MlxRealtimeSessionBranch,
    RealtimeModelKind,
};
pub use crate::composition::mlx::speculative::{MlxDrafter, MtpComponentTimingGuard};
pub use crate::composition::mlx::{
    inspect_model, MlxInspectionOptions, MlxModelInput, MlxModelOutput, MlxModelSession,
    MlxSessionCompletion,
};

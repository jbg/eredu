//! Curated application-facing MLX adapter API.

pub use crate::backend::mlx::error::Error as MlxError;
pub use crate::backend::mlx::nn::generation::sample;
pub use crate::backend::mlx::runtime::checkpoint::quantization::{
    quantize_checkpoint, CheckpointQuantizationOptions, CheckpointQuantizationReport,
};
pub use crate::backend::mlx::runtime::generation::sampler::{
    DefaultSampler, MirostatV2Sampler, Sampler,
};
pub use crate::backend::mlx::runtime::media::input::{InputPart, ModelInput};
pub use crate::backend::mlx::runtime::residency::expert_cache::{
    ExpertCacheReport, ExpertPassStatistics, ExpertTierStatistics,
};
pub use crate::backend::mlx::{
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

//! Curated application-facing MLX adapter API.

pub use crate::backend::error::Error as MlxError;
pub use crate::backend::runtime::checkpoint::quantization::{
    quantize_checkpoint, CheckpointQuantizationOptions, CheckpointQuantizationReport,
};
pub use crate::backend::runtime::residency::expert_cache::{
    ExpertCacheReport, ExpertPassStatistics, ExpertTierStatistics,
};
pub use crate::backend::{
    DeviceAssignment, MlxBackend, MlxModel, MlxModelConfig, MlxParallelContext, ModelLoadOptions,
};
pub use crate::composition::mlx::automatic::{
    create_realtime_backend, discover_hardware, expert_cache_telemetry, mtp_telemetry,
    residency_telemetry, MlxBackendFactory,
};
pub use crate::composition::mlx::speculative::MtpComponentTimingGuard;
pub use crate::composition::mlx::{inspect_model, MlxInspectionOptions, MlxModelOutput};

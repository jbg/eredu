//! Curated application-facing MLX adapter API.

pub use crate::composition::mlx::automatic::{
    create_realtime_backend, discover_hardware, expert_cache_telemetry, mtp_telemetry,
    residency_telemetry, MlxBackendFactory, MlxRealtimeAdapter,
};
pub use crate::composition::mlx::speculative::MtpComponentTimingGuard;

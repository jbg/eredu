//! Curated application-facing MLX adapter API.

use crate::backend::error::Error;
use eredu_core::residency::AllocatorMemoryMetrics;

pub use crate::composition::mlx::automatic::{
    create_realtime_backend, discover_hardware, expert_cache_telemetry, residency_telemetry,
    speculative_decoding_telemetry, MlxBackendFactory, MlxRealtimeAdapter,
};
pub use crate::composition::mlx::speculative::SpeculativeComponentTimingGuard;

/// Sets the process-global MLX allocator cache limit.
pub fn set_allocator_cache_limit(bytes: usize) -> Result<(), Error> {
    safemlx::memory::set_cache_limit(bytes)
        .map(|_| ())
        .map_err(Into::into)
}

/// Resets the process-global MLX allocator high-water mark.
pub fn reset_allocator_peak() -> Result<(), Error> {
    safemlx::memory::reset_peak_memory().map_err(Into::into)
}

/// Samples process-global MLX allocator memory.
pub fn allocator_memory() -> Result<AllocatorMemoryMetrics, Error> {
    crate::backend::residency::sample_allocator_memory().map_err(Into::into)
}

/// Overrides the Metal library path used by the MLX runtime.
#[cfg(feature = "metal")]
pub fn set_accelerator_library_path(path: impl AsRef<std::path::Path>) -> Result<(), Error> {
    safemlx::metal::set_metallib_path(path).map_err(Into::into)
}

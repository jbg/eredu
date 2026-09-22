//! Curated application-facing MLX adapter API.

use crate::backend::error::Error;
use eredu_core::residency::AllocatorMemoryMetrics;

pub use crate::composition::mlx::automatic::{
    create_realtime_execution, discover_hardware, parameter_bank_telemetry, MlxBackendFactory,
};
pub use crate::composition::mlx::speculative::SpeculativeComponentTimingGuard;

/// Sets the process-global MLX allocator cache limit and returns its previous value.
pub fn set_allocator_cache_limit(bytes: usize) -> Result<usize, Error> {
    safemlx::memory::set_cache_limit(bytes).map_err(Into::into)
}

/// Reads the current allocator cache limit without changing it or evicting cache.
/// This point-in-time query may initialize the native allocator.
pub fn allocator_cache_limit() -> Result<usize, Error> {
    safemlx::memory::cache_limit().map_err(Into::into)
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
#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub fn set_accelerator_library_path(path: impl AsRef<std::path::Path>) -> Result<(), Error> {
    safemlx::metal::set_metallib_path(path).map_err(Into::into)
}

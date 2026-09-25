//! Curated application-facing MLX adapter API.

use crate::backend::error::Error;
use eredu_core::residency::AllocatorMemoryMetrics;

pub use crate::composition::mlx::automatic::{
    create_realtime_execution, discover_hardware, parameter_bank_telemetry, MlxBackendFactory,
};
pub use crate::composition::mlx::speculative::SpeculativeComponentTimingGuard;

pub use crate::composition::mlx::inspection_mechanisms;

/// Sets the process-global MLX allocator cache limit and returns its previous value.
pub fn set_allocator_cache_limit(bytes: usize) -> Result<usize, Error> {
    safemlx::memory::set_cache_limit(bytes).map_err(Into::into)
}

/// Reads the current allocator cache limit without changing it or evicting cache.
/// This point-in-time query may initialize the native allocator.
pub fn allocator_cache_limit() -> Result<usize, Error> {
    safemlx::memory::cache_limit().map_err(Into::into)
}

/// Eredu's default cache ceiling. Smaller native defaults remain smaller.
pub const DEFAULT_ALLOCATOR_CACHE_LIMIT: usize = 256 * 1024 * 1024;

fn cache_policy_report(
    policy: safemlx::memory::CacheLimitPolicy,
) -> eredu_core::AllocatorCachePolicyReport {
    use eredu_core::AllocatorCachePolicySource as Source;
    use safemlx::memory::CacheLimitSource as Native;
    eredu_core::AllocatorCachePolicyReport {
        limit_bytes: policy.bytes as u64,
        source: match policy.source {
            Native::NativeDefault => Source::NativeDefault,
            Native::ManagedDefault => Source::ManagedDefault,
            Native::Explicit => Source::Explicit,
            Native::Preserved => Source::Preserved,
        },
    }
}

/// Reads the limit and its native provenance without mutation or eviction.
pub fn allocator_cache_policy() -> Result<eredu_core::AllocatorCachePolicyReport, Error> {
    safemlx::memory::cache_policy()
        .map(cache_policy_report)
        .map_err(Into::into)
}

/// Initializes the process-global allocator policy only if untouched. Explicit
/// native setters always win; the first initialization selects automatic or
/// preserved policy for subsequent sessions. Existing cache is not evicted.
pub fn initialize_allocator_cache_policy(
    preserve: bool,
) -> Result<eredu_core::AllocatorCachePolicyReport, Error> {
    safemlx::memory::configure_default_cache_policy(DEFAULT_ALLOCATOR_CACHE_LIMIT, preserve)
        .map(cache_policy_report)
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
#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub fn set_accelerator_library_path(path: impl AsRef<std::path::Path>) -> Result<(), Error> {
    safemlx::metal::set_metallib_path(path).map_err(Into::into)
}

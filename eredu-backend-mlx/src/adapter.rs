//! Curated application-facing MLX adapter API.

use crate::backend::error::Error;
use eredu_core::residency::AllocatorMemoryMetrics;

pub use crate::composition::mlx::automatic::{discover_hardware, MlxBackendFactory};

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

/// Preserves the untouched native default without changing its limit or evicting storage.
pub fn preserve_allocator_cache_policy() -> Result<eredu_core::AllocatorCachePolicyReport, Error> {
    safemlx::memory::configure_default_cache_policy(0, true)
        .map(cache_policy_report)
        .map_err(Into::into)
}

/// Native operations for portable initialization of the MLX cache ceiling.
pub struct MlxAllocatorCacheDefaults;

impl eredu_core::capability::NativeAllocatorCacheDefaults for MlxAllocatorCacheDefaults {
    fn initial_cache_facts(
        &self,
    ) -> Result<
        Option<eredu_core::capability::InitialAllocatorCacheFacts>,
        eredu_core::BackendFailure,
    > {
        let current = allocator_cache_policy().map_err(eredu_core::BackendFailure::from_error)?;
        Ok(
            (current.source == eredu_core::AllocatorCachePolicySource::NativeDefault).then(|| {
                eredu_core::capability::InitialAllocatorCacheFacts {
                    available_memory_bytes: crate::composition::mlx::capability::available_memory()
                        .ok()
                        .and_then(|memory| memory.available_memory_bytes.value().copied()),
                }
            }),
        )
    }

    fn apply_initial_cache_ceiling(&self, ceiling: u64) -> Result<(), eredu_core::BackendFailure> {
        let ceiling = usize::try_from(ceiling).map_err(eredu_core::BackendFailure::from_error)?;
        // The native lock remains authoritative if another process-local setter
        // wins while portable sizing is observing available memory.
        safemlx::memory::configure_default_cache_policy(ceiling, false)
            .map(|_| ())
            .map_err(eredu_core::BackendFailure::from_error)
    }
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

/// Releases unused allocator buffers at an explicit ordinary-host boundary.
/// Active tensors and unresolved completion roots remain owned; this does not
/// synchronize work, change cache policy, or release compiled functions.
pub fn trim_allocator_cache() -> Result<(), Error> {
    crate::backend::submission_recovery::reap();
    safemlx::reclaim_terminal_submission_resources()?;
    crate::backend::ordinary_retirement::reclaim_all();
    safemlx::memory::clear_cache().map_err(Into::into)
}

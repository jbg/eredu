//! MLX observations for backend-neutral residency telemetry.

use safemlx::error::Exception;
use safemlx_lm_core::residency::AllocatorMemoryMetrics;

/// Samples the MLX allocator without exposing MLX through the core contract.
pub fn sample_allocator_memory() -> Result<AllocatorMemoryMetrics, Exception> {
    let active = safemlx::memory::active_memory()?;
    let cached = safemlx::memory::cache_memory()?;
    let peak = safemlx::memory::peak_memory()?;
    Ok(AllocatorMemoryMetrics::new(
        u64::try_from(active).unwrap_or(u64::MAX),
        u64::try_from(cached).unwrap_or(u64::MAX),
        u64::try_from(peak).unwrap_or(u64::MAX),
    ))
}

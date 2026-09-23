//! MLX allocator memory statistics.
//!
//! These counters cover memory managed by MLX. They do not include all process
//! resident memory, memory-mapped checkpoint pages, or allocations owned by
//! unrelated libraries.

use crate::{
    error::{self, Result},
    utils::runtime_lock,
};

fn check_status(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(error::get_and_clear_last_mlx_error()
            .expect("MLX memory operation failed but no error was set")
            .into())
    }
}

/// Returns bytes currently held by active MLX allocations.
pub fn active_memory() -> Result<usize> {
    let _guard = runtime_lock::enter();
    error::ensure_mlx_error_handler();
    let mut bytes = 0;
    check_status(unsafe { safemlx_sys::mlx_get_active_memory(&mut bytes) })?;
    Ok(bytes)
}

/// Returns bytes currently retained by the MLX allocation cache.
pub fn cache_memory() -> Result<usize> {
    let _guard = runtime_lock::enter();
    error::ensure_mlx_error_handler();
    let mut bytes = 0;
    check_status(unsafe { safemlx_sys::mlx_get_cache_memory(&mut bytes) })?;
    Ok(bytes)
}

/// Returns the peak number of active MLX allocation bytes since the last reset.
pub fn peak_memory() -> Result<usize> {
    let _guard = runtime_lock::enter();
    error::ensure_mlx_error_handler();
    let mut bytes = 0;
    check_status(unsafe { safemlx_sys::mlx_get_peak_memory(&mut bytes) })?;
    Ok(bytes)
}

/// Resets the MLX peak-active-memory counter to the current active allocation.
pub fn reset_peak_memory() -> Result<()> {
    let _guard = runtime_lock::enter();
    error::ensure_mlx_error_handler();
    check_status(unsafe { safemlx_sys::mlx_reset_peak_memory() })
}

/// Releases allocations currently retained by MLX's allocator cache.
///
/// This is the allocator cache, not the separate compiled-function cache.
/// The operation affects process-global MLX state. It controls only memory
/// managed by MLX; it does not directly release or measure process RSS,
/// memory-mapped files, or unrelated native allocations.
pub fn clear_cache() -> Result<()> {
    let _guard = runtime_lock::enter();
    error::ensure_mlx_error_handler();
    check_status(unsafe { safemlx_sys::mlx_clear_cache() })
}

/// Sets the process-global MLX allocator-cache limit, in bytes.
///
/// Returns the previous limit reported by MLX. This controls only MLX-managed
/// allocations; it does not directly constrain process RSS, memory-mapped
/// files, or unrelated native allocations.
pub fn set_cache_limit(bytes: usize) -> Result<usize> {
    let _guard = runtime_lock::enter();
    error::ensure_mlx_error_handler();
    let mut previous = 0;
    check_status(unsafe { safemlx_sys::mlx_set_cache_limit(&mut previous, bytes) })?;
    Ok(previous)
}

/// Reads the current process-global MLX allocator-cache limit in bytes.
///
/// This may initialize the native allocator. It neither changes the limit nor
/// evicts cached allocations. The value is a snapshot; another caller may change
/// the limit later. Lowering the limit need not immediately evict existing cache.
pub fn cache_limit() -> Result<usize> {
    let _guard = runtime_lock::enter();
    error::ensure_mlx_error_handler();
    let mut bytes = 0;
    check_status(unsafe { safemlx_sys::mlx_get_cache_limit(&mut bytes) })?;
    Ok(bytes)
}

/// Native provenance of the process-global cache limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLimitSource {
    /// No setter or runtime policy has configured this allocator.
    NativeDefault,
    /// A runtime supplied a ceiling for the untouched native default.
    ManagedDefault,
    /// An explicit setter was called, even if it supplied the existing value.
    Explicit,
    /// Runtime initialization explicitly preserved the native default.
    Preserved,
}

/// Atomically observed native cache limit and its provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheLimitPolicy {
    /// Current limit, in bytes; existing cached allocations may exceed it.
    pub bytes: usize,
    /// How the current policy was selected.
    pub source: CacheLimitSource,
}

fn cache_policy_result(bytes: usize, source: i32) -> Result<CacheLimitPolicy> {
    let source = match source {
        0 => CacheLimitSource::NativeDefault,
        1 => CacheLimitSource::ManagedDefault,
        2 => CacheLimitSource::Explicit,
        3 => CacheLimitSource::Preserved,
        _ => return Err(error::Exception::from("unknown native cache policy source")),
    };
    Ok(CacheLimitPolicy { bytes, source })
}

/// Reads the native limit and provenance under one allocator lock. No mutation
/// or eviction occurs. Direct native setters are reflected in this observation.
pub fn cache_policy() -> Result<CacheLimitPolicy> {
    let _guard = runtime_lock::enter();
    error::ensure_mlx_error_handler();
    let (mut bytes, mut source) = (0, 0);
    check_status(unsafe { safemlx_sys::mlx_get_cache_policy(&mut bytes, &mut source) })?;
    cache_policy_result(bytes, source)
}

/// Configures only an untouched native default, atomically with native setters.
/// Uses the smaller of the existing limit and `ceiling`, or leaves it unchanged
/// when `preserve` is true. First initialization wins; explicit setters always
/// take precedence. Existing cache is not evicted. This is process-global.
pub fn configure_default_cache_policy(ceiling: usize, preserve: bool) -> Result<CacheLimitPolicy> {
    let _guard = runtime_lock::enter();
    error::ensure_mlx_error_handler();
    let (mut bytes, mut source) = (0, 0);
    check_status(unsafe {
        safemlx_sys::mlx_configure_default_cache_policy(&mut bytes, &mut source, ceiling, preserve)
    })?;
    cache_policy_result(bytes, source)
}

/// Sets the process-global MLX memory limit, in bytes.
///
/// Returns the previous limit reported by MLX. This limit applies to
/// MLX-managed allocations, not total process RSS, memory-mapped files, or
/// unrelated native allocations.
pub fn set_memory_limit(bytes: usize) -> Result<usize> {
    let _guard = runtime_lock::enter();
    error::ensure_mlx_error_handler();
    let mut previous = 0;
    check_status(unsafe { safemlx_sys::mlx_set_memory_limit(&mut previous, bytes) })?;
    Ok(previous)
}

/// Sets the process-global MLX wired-memory limit, in bytes.
///
/// Returns the previous limit reported by MLX. Wired-memory behavior and
/// support are backend- and platform-specific. This controls only
/// MLX-managed allocations; it does not directly constrain process RSS,
/// memory-mapped files, or unrelated native allocations.
pub fn set_wired_limit(bytes: usize) -> Result<usize> {
    let _guard = runtime_lock::enter();
    error::ensure_mlx_error_handler();
    let mut previous = 0;
    check_status(unsafe { safemlx_sys::mlx_set_wired_limit(&mut previous, bytes) })?;
    Ok(previous)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_headless_metal_error(error: &crate::error::Exception) -> bool {
        cfg!(feature = "metal") && error.what().contains("No Metal device available")
    }

    fn exercise_limit(set_limit: fn(usize) -> Result<usize>, temporary_limit: usize) {
        // Holding the same reentrant runtime lock used by MLX calls prevents
        // other lock-following unit tests from observing the temporary value.
        let _guard = runtime_lock::enter();
        let previous = match set_limit(temporary_limit) {
            Ok(previous) => previous,
            Err(error) if is_headless_metal_error(&error) => {
                // Headless macOS cannot initialize MLX's allocator. Requiring
                // this exact propagated error ensures the wrapper does not
                // convert an unsupported backend into false success.
                return;
            }
            Err(error) => panic!("setting the test limit failed unexpectedly: {error}"),
        };
        let test_result = std::panic::catch_unwind(|| {
            // MLX may normalize a requested limit. Round-trip the actual value
            // it reports and verify that the wrapper preserves the native result.
            let temporary =
                set_limit(previous).expect("restoring the original limit should succeed");
            let observed_original =
                set_limit(temporary).expect("reapplying the temporary limit should succeed");
            assert_eq!(observed_original, previous);
            set_limit(previous).expect("restoring the original limit should succeed");
        });

        match test_result {
            Ok(_) => {}
            Err(payload) => {
                if let Err(error) = set_limit(previous) {
                    panic!("failed to restore the original limit after a test panic: {error}");
                }
                std::panic::resume_unwind(payload);
            }
        }
    }

    #[test]
    fn allocator_cache_can_be_cleared() {
        match clear_cache() {
            Ok(()) => {}
            Err(error) if is_headless_metal_error(&error) => {
                assert!(error
                    .what()
                    .contains("safemlx-sys/src/mlx-c/mlx/c/memory.cpp"));
            }
            Err(error) => panic!("clearing the MLX allocator cache failed unexpectedly: {error}"),
        }
    }

    #[test]
    fn allocator_cache_limit_query_reads_native_state_without_eviction() {
        let _guard = runtime_lock::enter();
        let original = match cache_limit() {
            Ok(limit) => limit,
            Err(error) if is_headless_metal_error(&error) => return,
            Err(error) => panic!("querying allocator cache limit failed: {error}"),
        };
        let result = std::panic::catch_unwind(|| {
            assert_eq!(set_cache_limit(1024 * 1024).unwrap(), original);
            let array = crate::Array::from_slice(&vec![1.0f32; 4096], &[4096]);
            drop(array);
            let retained = cache_memory().unwrap();
            assert!(retained > 0, "fixture must populate the allocator cache");
            assert_eq!(cache_limit().unwrap(), 1024 * 1024);
            assert_eq!(cache_limit().unwrap(), 1024 * 1024);
            assert_eq!(cache_memory().unwrap(), retained);

            // A native caller can change policy without using our Rust setter.
            // The getter must observe it rather than a shadow Rust value.
            let mut previous = 0;
            check_status(unsafe { safemlx_sys::mlx_set_cache_limit(&mut previous, 0) }).unwrap();
            assert_eq!(previous, 1024 * 1024);
            assert_eq!(cache_limit().unwrap(), 0);
            assert_eq!(set_cache_limit(original).unwrap(), 0);
            assert_eq!(cache_limit().unwrap(), original);
        });
        set_cache_limit(original).expect("restore allocator policy after test");
        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
    }

    #[test]
    fn allocator_limits_return_and_restore_previous_values() {
        exercise_limit(set_cache_limit, usize::MAX);
        exercise_limit(set_memory_limit, usize::MAX);
        // MLX rejects wired limits above the device's maximum working set.
        // Zero is a valid temporary limit and the runtime lock prevents other
        // lock-following tests from observing it before restoration.
        exercise_limit(set_wired_limit, 0);
    }
}

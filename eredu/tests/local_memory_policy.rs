//! Native local-adapter policy coverage, separate from portable conformance.
#![cfg(feature = "mlx")]

use eredu::api::{
    local_allocator_cache_limit, set_local_allocator_cache_limit, GenerationMemoryOptions,
    GenerationMemoryPlacement,
};
use eredu_core::InputTokenCount;

#[test]
fn local_forecast_observes_and_restores_allocator_cache_policy() {
    let original = match local_allocator_cache_limit() {
        Ok(limit) => limit,
        Err(error)
            if cfg!(feature = "metal")
                && error.to_string().contains("No Metal device available") =>
        {
            return
        }
        Err(error) => panic!("allocator policy query failed: {error}"),
    };
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            set_local_allocator_cache_limit(self.0).expect("restore original cache limit");
        }
    }
    let _restore = Restore(original);
    let input = InputTokenCount::text(16);
    let initial =
        GenerationMemoryOptions::for_local_backend(input, GenerationMemoryPlacement::Host);
    assert!(initial.backend_overhead.upper_bytes.is_some());
    assert_eq!(local_allocator_cache_limit().unwrap(), original);
    assert_eq!(set_local_allocator_cache_limit(0).unwrap(), original);
    let zero = GenerationMemoryOptions::for_local_backend(input, GenerationMemoryPlacement::Host);
    assert_eq!(zero.backend_overhead.upper_bytes, Some(64 * 1024 * 1024));
    assert_eq!(set_local_allocator_cache_limit(1024 * 1024).unwrap(), 0);
    let changed =
        GenerationMemoryOptions::for_local_backend(input, GenerationMemoryPlacement::Host);
    assert_eq!(changed.backend_overhead.upper_bytes, Some(65 * 1024 * 1024));
    assert_eq!(local_allocator_cache_limit().unwrap(), 1024 * 1024);
    // Backend-neutral construction remains independent of native observations.
    assert_eq!(
        GenerationMemoryOptions::new(input, GenerationMemoryPlacement::Host)
            .backend_overhead
            .upper_bytes,
        None
    );
    assert_eq!(
        set_local_allocator_cache_limit(original).unwrap(),
        1024 * 1024
    );
    assert_eq!(local_allocator_cache_limit().unwrap(), original);
}

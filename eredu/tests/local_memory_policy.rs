//! Native local-adapter policy coverage, separate from portable conformance.
#![cfg(feature = "mlx")]

use eredu::api::{GenerationMemoryOptions, GenerationMemoryPlacement};
use eredu_backend_mlx::{
    allocator_cache_limit, allocator_cache_policy, initialize_allocator_cache_policy,
    set_allocator_cache_limit,
};
use eredu_core::InputTokenCount;

#[test]
fn runtime_default_policy_and_forecast_observation() {
    use eredu_core::AllocatorCachePolicySource as Source;
    let Ok(scenario) = std::env::var("EREDU_LOCAL_CACHE_POLICY_SCENARIO") else {
        for scenario in ["automatic", "preserved", "explicit", "disabled"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "runtime_default_policy_and_forecast_observation",
                    "--nocapture",
                ])
                .env("EREDU_LOCAL_CACHE_POLICY_SCENARIO", scenario)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{scenario}: {}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        return;
    };
    let initial = match allocator_cache_policy() {
        Ok(policy) => policy,
        Err(error)
            if cfg!(feature = "metal")
                && error.to_string().contains("No Metal device available") =>
        {
            return
        }
        Err(error) => panic!("cache policy query failed: {error}"),
    };
    assert_eq!(initial.source, Source::NativeDefault);
    let input = InputTokenCount::text(16);
    let before = GenerationMemoryOptions::for_backend(
        &eredu_backend_mlx::MlxBackendFactory::default(),
        input,
        GenerationMemoryPlacement::Host,
    );
    assert_eq!(allocator_cache_policy().unwrap(), initial);
    assert!(before.backend_overhead.detail.contains("NativeDefault"));
    let cpu = eredu_core::DevicePlan::new("mlx", "cpu:0").unwrap();
    let device_options = GenerationMemoryOptions::for_backend_device(
        &eredu_backend_mlx::MlxBackendFactory::default(),
        input,
        &cpu,
    )
    .unwrap();
    assert!(matches!(
        device_options.placement,
        GenerationMemoryPlacement::Host
    ));
    assert_eq!(
        device_options.budget.available_bytes,
        device_options.host_budget.available_bytes
    );
    assert_eq!(device_options.budget.application_limit_bytes, None);
    assert_eq!(device_options.budget.reserve_bytes, 0);
    assert!(device_options
        .backend_overhead
        .detail
        .contains("NativeDefault"));
    assert_eq!(allocator_cache_policy().unwrap(), initial);
    match scenario.as_str() {
        "automatic" => {
            initialize_allocator_cache_policy(false).unwrap();
        }
        "preserved" => {
            initialize_allocator_cache_policy(true).unwrap();
        }
        "explicit" => {
            set_allocator_cache_limit(initial.limit_bytes as usize).unwrap();
        }
        "disabled" => {
            set_allocator_cache_limit(0).unwrap();
        }
        _ => panic!("unknown scenario"),
    }
    let configured = allocator_cache_policy().unwrap();
    match scenario.as_str() {
        "automatic" => {
            assert_eq!(configured.source, Source::ManagedDefault);
            // Available memory is observed at initialization, so its exact
            // value can change between calls. Portable sizing tests cover the
            // fraction; this integration checks native caps and ownership.
            assert!(configured.limit_bytes <= initial.limit_bytes.min(8 * 1024 * 1024 * 1024));
            assert_eq!(
                serde_json::to_value(configured).unwrap()["source"],
                "managed_default"
            );
        }
        "preserved" => {
            assert_eq!(configured.source, Source::Preserved);
            assert_eq!(configured.limit_bytes, initial.limit_bytes);
        }
        "disabled" => {
            assert_eq!(configured.source, Source::Explicit);
            assert_eq!(configured.limit_bytes, 0);
        }
        _ => {
            assert_eq!(configured.source, Source::Explicit);
            assert_eq!(configured.limit_bytes, initial.limit_bytes);
        }
    }
    initialize_allocator_cache_policy(false).unwrap();
    assert_eq!(allocator_cache_policy().unwrap(), configured);
    let forecast = GenerationMemoryOptions::for_backend(
        &eredu_backend_mlx::MlxBackendFactory::default(),
        input,
        GenerationMemoryPlacement::Host,
    );
    assert_eq!(
        forecast.backend_overhead.upper_bytes,
        Some(configured.limit_bytes + 64 * 1024 * 1024)
    );
    assert_eq!(allocator_cache_policy().unwrap(), configured);
}

#[test]
fn local_forecast_rejects_invalid_or_undiscovered_devices() {
    for (backend, device) in [
        ("other", "cpu:0"),
        ("mlx", "cpu:00"),
        ("mlx", "gpu:0"),
        ("mlx", "cpu:1234"),
    ] {
        let error = GenerationMemoryOptions::for_backend_device(
            &eredu_backend_mlx::MlxBackendFactory::default(),
            InputTokenCount::text(16),
            &eredu_core::DevicePlan::new(backend, device).unwrap(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            eredu::api::GenerationForecastError::Capability(
                eredu_core::CapabilityError::InvalidConfiguration { .. }
            )
        ));
    }
}

#[test]
fn local_forecast_observes_and_restores_allocator_cache_policy() {
    let original = match allocator_cache_limit() {
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
            set_allocator_cache_limit(self.0).expect("restore original cache limit");
        }
    }
    let _restore = Restore(original);
    let input = InputTokenCount::text(16);
    let initial = GenerationMemoryOptions::for_backend(
        &eredu_backend_mlx::MlxBackendFactory::default(),
        input,
        GenerationMemoryPlacement::Host,
    );
    assert!(initial.backend_overhead.upper_bytes.is_some());
    assert_eq!(allocator_cache_limit().unwrap(), original);
    assert_eq!(set_allocator_cache_limit(0).unwrap(), original);
    let zero = GenerationMemoryOptions::for_backend(
        &eredu_backend_mlx::MlxBackendFactory::default(),
        input,
        GenerationMemoryPlacement::Host,
    );
    assert_eq!(zero.backend_overhead.upper_bytes, Some(64 * 1024 * 1024));
    assert_eq!(set_allocator_cache_limit(1024 * 1024).unwrap(), 0);
    let changed = GenerationMemoryOptions::for_backend(
        &eredu_backend_mlx::MlxBackendFactory::default(),
        input,
        GenerationMemoryPlacement::Host,
    );
    assert_eq!(changed.backend_overhead.upper_bytes, Some(65 * 1024 * 1024));
    assert_eq!(allocator_cache_limit().unwrap(), 1024 * 1024);
    // Backend-neutral construction remains independent of native observations.
    assert_eq!(
        GenerationMemoryOptions::new(input, GenerationMemoryPlacement::Host)
            .backend_overhead
            .upper_bytes,
        None
    );
    assert_eq!(set_allocator_cache_limit(original).unwrap(), 1024 * 1024);
    assert_eq!(allocator_cache_limit().unwrap(), original);
}

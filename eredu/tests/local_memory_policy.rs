//! Native local-adapter policy coverage, separate from portable conformance.
#![cfg(feature = "mlx")]

use eredu::api::{
    local_allocator_cache_limit, set_local_allocator_cache_limit, GenerationMemoryOptions,
    GenerationMemoryPlacement,
};
use eredu_core::InputTokenCount;

#[test]
fn runtime_default_policy_and_forecast_observation() {
    use eredu::api::{
        configure_local_runtime, local_allocator_cache_policy,
        AllocatorCachePolicySource as Source, LocalAllocatorCachePolicy, LocalRuntimeConfiguration,
    };
    let Ok(scenario) = std::env::var("EREDU_LOCAL_CACHE_POLICY_SCENARIO") else {
        for scenario in ["automatic", "preserved", "explicit"] {
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
    let initial = match local_allocator_cache_policy() {
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
    let before = GenerationMemoryOptions::for_local_backend(input, GenerationMemoryPlacement::Host);
    assert_eq!(local_allocator_cache_policy().unwrap(), initial);
    assert!(before.backend_overhead.detail.contains("NativeDefault"));
    let cpu = eredu::api::local_device_plan(eredu::api::LocalDevice::Cpu).unwrap();
    let device_options = GenerationMemoryOptions::for_local_device(input, &cpu).unwrap();
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
    assert_eq!(local_allocator_cache_policy().unwrap(), initial);
    let configuration = match scenario.as_str() {
        "automatic" => LocalRuntimeConfiguration::default(),
        "preserved" => LocalRuntimeConfiguration::default()
            .with_allocator_cache_policy(LocalAllocatorCachePolicy::PreserveNative),
        "explicit" => LocalRuntimeConfiguration::default()
            .with_allocator_cache_limit(initial.limit_bytes as usize),
        _ => panic!("unknown scenario"),
    };
    configure_local_runtime(&configuration).unwrap();
    let configured = local_allocator_cache_policy().unwrap();
    match scenario.as_str() {
        "automatic" => {
            assert_eq!(configured.source, Source::ManagedDefault);
            assert_eq!(
                configured.limit_bytes,
                initial.limit_bytes.min(256 * 1024 * 1024)
            );
            assert_eq!(
                serde_json::to_value(configured).unwrap()["source"],
                "managed_default"
            );
        }
        "preserved" => {
            assert_eq!(configured.source, Source::Preserved);
            assert_eq!(configured.limit_bytes, initial.limit_bytes);
        }
        _ => {
            assert_eq!(configured.source, Source::Explicit);
            assert_eq!(configured.limit_bytes, initial.limit_bytes);
        }
    }
    configure_local_runtime(&LocalRuntimeConfiguration::default()).unwrap();
    assert_eq!(local_allocator_cache_policy().unwrap(), configured);
    let forecast =
        GenerationMemoryOptions::for_local_backend(input, GenerationMemoryPlacement::Host);
    assert_eq!(
        forecast.backend_overhead.upper_bytes,
        Some(configured.limit_bytes + 64 * 1024 * 1024)
    );
    assert_eq!(local_allocator_cache_policy().unwrap(), configured);
}

#[test]
fn local_forecast_rejects_invalid_or_undiscovered_devices() {
    for (backend, device) in [
        ("other", "cpu:0"),
        ("mlx", "cpu:00"),
        ("mlx", "gpu:0"),
        ("mlx", "cpu:1234"),
    ] {
        let error = GenerationMemoryOptions::for_local_device(
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

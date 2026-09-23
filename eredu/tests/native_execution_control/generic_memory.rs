use super::*;

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run explicitly with an accessible Metal device"
)]
fn native_generic_workspace_covers_softcapped_and_routed_modules() {
    let previous = set_local_allocator_cache_limit(0).unwrap();
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            set_local_allocator_cache_limit(self.0).unwrap();
        }
    }
    let _restore = Restore(previous);
    let device = if cfg!(feature = "metal") {
        LocalDevice::Accelerator(0)
    } else {
        LocalDevice::Cpu
    };
    let gemma = fixture(false);
    use_family_weights(&gemma.0, "gemma2");
    for (name, root) in [
        ("gemma2", gemma),
        ("qwen3_moe", routed_components::routed_fixture()),
    ] {
        let plan = ExecutionPlan::fully_resident(local_device_plan(device).unwrap());
        let (mut model, _) =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &plan)
                .unwrap()
                .into_parts();
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(8),
                temperature: Some(0.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let options = GenerationForecastOptions {
            budget: MemoryBudget {
                application_limit_bytes: Some(1 << 30),
                ..Default::default()
            },
            ..Default::default()
        };
        let ids = vec![1; 39];
        model.synchronize().unwrap();
        let baseline = eredu_backend_mlx::allocator_memory()
            .unwrap()
            .active_bytes();
        let forecast = model.forecast_token_ids(&ids, settings, &options).unwrap();
        assert_eq!(forecast.estimate.fit, MemoryFit::LikelyFit);
        assert!(forecast.request.domains[0].executions[0]
            .workspace
            .is_none());
        assert!(forecast.request.domains[0].executions[0]
            .execution_topology
            .is_some());
        assert_eq!(
            eredu_backend_mlx::allocator_memory()
                .unwrap()
                .active_bytes(),
            baseline
        );
        reset_local_allocator_peak().unwrap();
        let config = eredu_core::TextGenerationConfig::new(
            model.resolve_generation_config(settings.overrides).unwrap(),
        );
        let mut tokens = model.generate_tokens(ids, config).unwrap();
        for _ in 0..4 {
            assert!(tokens.next().unwrap().is_ok());
        }
        tokens.synchronize().unwrap();
        let first_peak = eredu_backend_mlx::allocator_memory().unwrap().peak_bytes();
        let continuation_baseline = eredu_backend_mlx::allocator_memory()
            .unwrap()
            .active_bytes();
        let continued = tokens.forecast_remaining_generation(4, &options).unwrap();
        assert_eq!(continued.estimate.fit, MemoryFit::LikelyFit);
        assert_eq!(
            eredu_backend_mlx::allocator_memory()
                .unwrap()
                .active_bytes(),
            continuation_baseline
        );
        reset_local_allocator_peak().unwrap();
        for token in tokens {
            token.unwrap();
        }
        model.synchronize().unwrap();
        let peak = eredu_backend_mlx::allocator_memory().unwrap().peak_bytes();
        let growth = first_peak.max(peak).saturating_sub(baseline);
        let continuation_growth = peak.saturating_sub(continuation_baseline);
        let upper = forecast.estimate.domains[0]
            .additional_generation_peak
            .upper_bytes
            .unwrap();
        let continuation_upper = continued.estimate.domains[0]
            .additional_generation_peak
            .upper_bytes
            .unwrap();
        assert!(growth <= upper, "{name}: growth {growth}, upper {upper}");
        assert!(
            continuation_growth <= continuation_upper,
            "{name}: continuation growth {continuation_growth}, upper {continuation_upper}"
        );
        eprintln!("generic workspace: {name}, growth={growth}, upper={upper}, continuation_growth={continuation_growth}, continuation_upper={continuation_upper}");
    }
}

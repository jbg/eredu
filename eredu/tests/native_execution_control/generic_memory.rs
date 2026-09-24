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
            calibration: ForecastCalibration {
                graph_driver_bytes: 0,
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
        let serialized: GenerationForecast =
            serde_json::from_str(&serde_json::to_string(&forecast).unwrap()).unwrap();
        assert_eq!(
            serialized
                .with_max_output_tokens(8)
                .unwrap()
                .estimate
                .domains,
            forecast.estimate.domains
        );
        let repeated = model.forecast_token_ids(&ids, settings, &options).unwrap();
        assert_eq!(
            repeated.estimate.domains[0].generation_peak,
            forecast.estimate.domains[0].generation_peak
        );
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
        let mut tokens = model.generate_tokens(ids.clone(), config).unwrap();
        let mut generated = Vec::new();
        for _ in 0..4 {
            generated.push(tokens.next().unwrap().unwrap().token_id().unwrap());
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
            generated.push(token.unwrap().token_id().unwrap());
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
        model.reset().unwrap();
        let reset_active = eredu_backend_mlx::allocator_memory()
            .unwrap()
            .active_bytes();
        let reset_forecast = model.forecast_token_ids(&ids, settings, &options).unwrap();
        assert_eq!(
            reset_forecast.estimate.domains[0].generation_peak,
            forecast.estimate.domains[0].generation_peak
        );
        assert_eq!(
            eredu_backend_mlx::allocator_memory()
                .unwrap()
                .active_bytes(),
            reset_active
        );
        let replay_config = eredu_core::TextGenerationConfig::new(
            model.resolve_generation_config(settings.overrides).unwrap(),
        );
        let replay: Vec<_> = model
            .generate_tokens(ids.clone(), replay_config)
            .unwrap()
            .map(|token| token.unwrap().token_id().unwrap())
            .collect();
        assert_eq!(
            replay, generated,
            "{name}: reset and forecast preserve cached generation"
        );
        model.reset().unwrap();
        let chat = model
            .prepare_chat(ChatTemplateRequest {
                messages: vec![serde_json::json!({"role":"user", "content":"left right"})],
                add_generation_prompt: true,
                ..Default::default()
            })
            .unwrap();
        let prepared = model
            .prepare_observed_token_ids(
                &chat,
                ids,
                settings,
                CapturePlan::none(),
                TraceLimits {
                    per_record_bytes: 16 << 10,
                    total_bytes: 256 << 10,
                },
            )
            .unwrap();
        let mut run = model
            .start_controlled_text(prepared, &[], Default::default(), |_| {
                ControlFlow::Continue(())
            })
            .unwrap();
        while run.finish_reason().is_none() {
            run.step(|_| ControlFlow::Continue(())).unwrap();
        }
        assert_eq!(
            run.token_ids(),
            generated,
            "{name}: controlled generation parity"
        );
        eprintln!("generic workspace: {name}, growth={growth}, upper={upper}, continuation_growth={continuation_growth}, continuation_upper={continuation_upper}, reset_and_controlled_parity=true");
    }
}

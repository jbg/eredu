use super::*;
use eredu_core::TextGenerationConfig;

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run explicitly with an accessible Metal device"
)]
fn native_continuation_forecasts_cover_dense_and_sliding_cache_growth() {
    let device = if cfg!(feature = "metal") {
        LocalDevice::Accelerator(0)
    } else {
        LocalDevice::Cpu
    };
    if let Some(path) = std::env::var_os("EREDU_CONTINUATION_MODEL") {
        check(Path::new(&path), device, true);
    } else {
        for family in ["qwen2", "qwen2_sliding", "gemma2"] {
            let root = fixture(false);
            if family == "qwen2_sliding" {
                let path = root.0.join("config.json");
                let mut config: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
                config["use_sliding_window"] = true.into();
                config["sliding_window"] = 2.into();
                config["max_window_layers"] = 1.into();
                std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
            } else {
                use_family_weights(&root.0, family);
            }
            check(&root.0, device, family != "gemma2");
        }
    }
}

fn check(path: &Path, device: LocalDevice, workspace_known: bool) {
    let expected_fit = if workspace_known {
        MemoryFit::LikelyFit
    } else {
        MemoryFit::InsufficientInformation
    };
    let plan = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
        .with_required_session_capabilities(SessionCapabilities::new(true, true, true));
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), path, &plan)
            .unwrap()
            .into_parts();
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user", "content":"hello hello hello hello"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let prompt = model.count_prepared_chat(&chat).unwrap().model_positions;
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(32),
            temperature: Some(0.0),
            ..Default::default()
        },
        ..Default::default()
    };
    let prepared = model
        .prepare_observed_chat(
            &chat,
            settings,
            CapturePlan::none(),
            TraceLimits {
                per_record_bytes: 16384,
                total_bytes: 64 << 20,
            },
        )
        .unwrap();
    let mut run = model
        .start_controlled_text(prepared, &[], Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    run.enable_snapshots(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 0,
        retained_bytes: 256 << 20,
        cumulative_copy_bytes: 1 << 30,
    })
    .unwrap();
    run.force_next_token(1).unwrap();
    run.step(|_| ControlFlow::Continue(())).unwrap();
    let saved = run.snapshot(|_| ControlFlow::Continue(())).unwrap();
    let usage = run.snapshot_usage();
    let before = eredu_backend_mlx::allocator_memory().unwrap();
    let forecast = run
        .forecast_remaining_generation(16, &Default::default())
        .unwrap();
    assert_eq!(forecast.continuation.current_positions, prompt);
    assert_eq!(forecast.estimate.requested_positions, prompt + 16);
    assert_eq!(forecast.estimate.fit, expected_fit);
    // Includes a retained encoded trace and a distinct semantic history. The
    // 64 MiB trace must not become several GiB through per-byte event headers.
    let retained_upper = forecast.request.domains[0]
        .retained_input
        .upper_bytes
        .unwrap();
    assert!(
        retained_upper < 256 << 20,
        "retained upper: {retained_upper}"
    );
    assert_eq!(run.snapshot_usage(), usage);
    assert_eq!(
        eredu_backend_mlx::allocator_memory()
            .unwrap()
            .active_bytes(),
        before.active_bytes()
    );
    reset_local_allocator_peak().unwrap();
    for _ in 0..16 {
        run.force_next_token(1).unwrap();
        run.step(|_| ControlFlow::Continue(())).unwrap();
    }
    let measured = eredu_backend_mlx::allocator_memory().unwrap();
    let upper = forecast.estimate.domains[0]
        .additional_generation_peak
        .upper_bytes;
    assert!(forecast.continuation.peak_state.upper_bytes.is_some());
    if let Some(upper) = upper {
        assert!(measured.peak_bytes().saturating_sub(before.active_bytes()) <= upper);
    }
    let advanced = run
        .forecast_remaining_generation(1, &Default::default())
        .unwrap();
    assert_eq!(advanced.continuation.current_positions, prompt + 16);
    run.restore(&saved, |_| ControlFlow::Continue(())).unwrap();
    let restored = run
        .forecast_remaining_generation(16, &Default::default())
        .unwrap();
    assert_eq!(restored.continuation.current_positions, prompt);
    assert_eq!(restored.estimate.fit, expected_fit);
    eprintln!(
        "{}: prefix {}, horizon 16, retained upper {}, additional upper {:?}, measured active growth {}",
        path.display(),
        prompt,
        retained_upper,
        upper,
        measured.peak_bytes().saturating_sub(before.active_bytes())
    );
    drop(run);
    model.reset().unwrap();
    let config = TextGenerationConfig::new(
        model
            .resolve_generation_config(GenerationConfigOverrides {
                max_new_tokens: Some(4),
                temperature: Some(0.0),
                ..Default::default()
            })
            .unwrap(),
    );
    let mut tokens = model.generate_tokens(vec![1; 17], config).unwrap();
    tokens.next().unwrap().unwrap().token_id().unwrap();
    tokens.synchronize().unwrap();
    let forecast = tokens
        .forecast_remaining_generation(3, &Default::default())
        .unwrap();
    assert_eq!(forecast.continuation.current_positions, 17);
    assert_eq!(forecast.estimate.fit, expected_fit);
    assert_eq!(tokens.count(), 3);
}

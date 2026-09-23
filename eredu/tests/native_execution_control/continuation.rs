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

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run explicitly with an accessible Metal device"
)]
fn native_capture_forecasts_project_top_k_geometry_and_continuations() {
    let fixture = fixture(false);
    let path = std::env::var_os("EREDU_CONTINUATION_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| fixture.0.clone());
    let device = if cfg!(feature = "metal") {
        LocalDevice::Accelerator(0)
    } else {
        LocalDevice::Cpu
    };
    let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap());
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &path, &execution)
            .unwrap()
            .into_parts();
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user", "content":"hello hello hello hello"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(32),
            temperature: Some(0.0),
            ..Default::default()
        },
        ..Default::default()
    };
    let usage = CaptureUsage {
        captures: 128,
        retained_bytes: 1 << 30,
        host_bytes: 1 << 30,
        encoded_bytes: 1 << 30,
    };
    let mut capture = CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selections: vec![CaptureSelection {
            id: "top-k".into(),
            path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
            schedule: Default::default(),
            slices: vec![],
            transform: CaptureTransform::TopCandidates { count: 8 },
        }],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    };
    let trace = TraceLimits {
        per_record_bytes: 64 << 10,
        total_bytes: 64 << 20,
    };
    let prepared = model
        .prepare_observed_chat(&chat, settings, capture.clone(), trace)
        .unwrap();
    let before = eredu_backend_mlx::allocator_memory().unwrap();
    let forecast = model
        .forecast_observed_generation(&prepared, &Default::default())
        .unwrap();
    let projection = forecast
        .capture
        .as_ref()
        .unwrap()
        .projection
        .as_ref()
        .unwrap();
    let projected = projection.outlook(0, 32).unwrap().unwrap();
    assert!(projected.complete);
    assert_eq!(projected.total.captures, 32);
    assert!(
        forecast.request.domains[0]
            .retained_input
            .upper_bytes
            .unwrap()
            < 72 << 20
    );
    assert_eq!(forecast.estimate.fit, MemoryFit::LikelyFit);
    capture.limits.per_step.retained_bytes *= 2;
    capture.limits.cumulative.host_bytes *= 2;
    let larger = model
        .prepare_observed_chat(&chat, settings, capture.clone(), trace)
        .unwrap();
    assert_eq!(
        model
            .forecast_observed_generation(&larger, &Default::default())
            .unwrap()
            .request
            .domains[0]
            .retained_input
            .upper_bytes,
        forecast.request.domains[0].retained_input.upper_bytes
    );
    capture.selections[0].transform = CaptureTransform::TopCandidates { count: 16 };
    let wider = model
        .prepare_observed_chat(&chat, settings, capture.clone(), trace)
        .unwrap();
    let wider = model
        .forecast_observed_generation(&wider, &Default::default())
        .unwrap();
    assert!(
        wider.request.domains[0].retained_input.upper_bytes
            > forecast.request.domains[0].retained_input.upper_bytes
    );
    capture.selections[0].transform = CaptureTransform::TopCandidates { count: 8 };
    capture.selections[0].schedule.every = 2;
    let sparse = model
        .prepare_observed_chat(&chat, settings, capture, trace)
        .unwrap();
    let sparse = model
        .forecast_observed_generation(&sparse, &Default::default())
        .unwrap();
    assert!(
        sparse.request.domains[0].retained_input.upper_bytes
            < forecast.request.domains[0].retained_input.upper_bytes
    );
    assert_eq!(
        eredu_backend_mlx::allocator_memory()
            .unwrap()
            .active_bytes(),
        before.active_bytes()
    );
    let mut run = model
        .start_controlled_text(prepared, &[], Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    let mut charged = CaptureUsage::default();
    for prediction in 0..4 {
        run.force_next_token(1).unwrap();
        run.step(|record| {
            if let ObservedGenerationEvent::Token {
                captures: Some(step),
                ..
            } = &record.generation.event
            {
                assert_eq!(step.records[0].outcome, CaptureOutcome::Captured);
                charged = step.cumulative_usage;
            }
            ControlFlow::Continue(())
        })
        .unwrap();
        let predicted = projection
            .outlook(0, prediction + 1)
            .unwrap()
            .unwrap()
            .total;
        assert!(
            charged.exceeded(predicted).is_none(),
            "charged {charged:?}, projected {predicted:?}"
        );
        let emitted = run.emitted_bytes();
        let remaining = run
            .forecast_remaining_generation(4, &Default::default())
            .unwrap();
        let capture = remaining.capture.as_ref().unwrap();
        assert_eq!(capture.first_prediction, prediction + 1);
        assert_eq!(capture.inherited_usage, charged);
        assert!(
            remaining.request.domains[0]
                .retained_input
                .upper_bytes
                .unwrap()
                < 144 << 20
        );
        assert_eq!(run.emitted_bytes(), emitted);
    }
    eprintln!("{}: top-k 8, horizon 32, 1 GiB capture ceilings, retained upper {}, projected host {}, projected native peak {:?}", path.display(), forecast.request.domains[0].retained_input.upper_bytes.unwrap(), projected.total.host_bytes, projected.phases.map(|p| p.retained_bytes));
}

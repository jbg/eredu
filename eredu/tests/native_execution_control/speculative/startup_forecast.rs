//! Startup projections through the public embedded speculative path.
use super::*;

pub(super) fn attention_prediction_fixture(mixed: bool) -> Fixture {
    let root = fixture(false);
    let config = serde_json::json!({
        "model_type":"qwen3_5_text", "hidden_size":8, "vocab_size":64,
        "num_hidden_layers":2, "num_attention_heads":2, "num_key_value_heads":1,
        "head_dim":4, "intermediate_size":16, "num_experts":0,
        "max_position_embeddings":128, "tie_word_embeddings":true,
        "layer_types":["full_attention","full_attention"],
        "mtp_num_hidden_layers":2, "eos_token_id":[]
    });
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    write_tensor_plan(&root.0, resolved.architecture.checkpoint());
    if mixed {
        // Keep F32 normalization with BF16 matrices to exercise native promotion caches.
        let source = std::fs::read(root.0.join("model.safetensors")).unwrap();
        let header_len = u64::from_le_bytes(source[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&source[8..8 + header_len]).unwrap();
        let tensors = header
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, tensor)| {
                let offsets = tensor["data_offsets"].as_array().unwrap();
                let data = &source[8 + header_len + offsets[0].as_u64().unwrap() as usize
                    ..8 + header_len + offsets[1].as_u64().unwrap() as usize];
                let (dtype, bytes) = if name.contains("norm") {
                    ("F32", data.to_vec())
                } else {
                    (
                        "BF16",
                        data.as_chunks::<4>()
                            .0
                            .iter()
                            .flat_map(|value| {
                                // Truncated upper bits define exact finite BF16 fixture values.
                                ((u32::from_le_bytes(*value) >> 16) as u16).to_le_bytes()
                            })
                            .collect(),
                    )
                };
                let shape = tensor["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_u64().unwrap() as usize)
                    .collect();
                (name.clone(), (dtype.into(), shape, bytes))
            })
            .collect();
        super::super::quantized_parameters::write_tensors(&root.0, &tensors);
    }
    std::fs::write(
        root.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    root
}

fn check_cold_forecast(root: &Path, execution: &ExecutionPlan, lookahead: bool) {
    let factory = MlxBackendFactory::default();
    let before = eredu_backend_mlx::allocator_memory()
        .unwrap()
        .active_bytes();
    let inspected = eredu_backend_mlx::native::inspect_model_preparation(
        root,
        eredu_backend_mlx::native::MlxInspectionOptions::new(
            factory.load_request_for_plan(execution).unwrap(),
        ),
    )
    .unwrap();
    let mut options = GenerationMemoryOptions::new(
        eredu_core::InputTokenCount::text(8),
        GenerationMemoryPlacement::Unified,
    );
    options.max_output_tokens = Some(8);
    options.attention = AttentionWorkspace::ScoreMatrixUpperBound;
    options.cache_update = CacheUpdateWorkspace::CopyState;
    options.workspace_overlap = WorkspaceOverlap {
        upper_live_copies: Some(4),
        detail: "explicit native test calibration".into(),
    };
    options.backend_overhead = MemoryBytes::exact(0);
    options.budget.application_limit_bytes = Some(1 << 30);
    let (request, plan) = inspected_speculative_generation_memory_plan(
        &inspected,
        &options,
        2,
        eredu_core::generation::SpeculativeSchedulerOptions::default().with_lookahead(lookahead),
        MemoryBytes::estimated(0, 128, "MLX sampling calibration for this native test"),
    )
    .unwrap();
    let estimate =
        eredu_runtime::memory_forecast::estimate_speculative_memory(&request, &plan).unwrap();
    assert_eq!(estimate.fit, MemoryFit::LikelyFit, "{estimate:?}");
    // The legacy ordinary request must not silently omit the prediction transaction.
    let legacy = inspected_generation_memory_request(&inspected, &options).unwrap();
    assert_eq!(
        estimate_generation_memory(&legacy).unwrap().fit,
        MemoryFit::InsufficientInformation
    );
    // Without explicit backend sampling facts the convenience estimate remains honest.
    let incomplete = estimate_inspected_generation_memory(&inspected, &options).unwrap();
    assert_eq!(incomplete.fit, MemoryFit::InsufficientInformation);
    assert_eq!(
        eredu_backend_mlx::allocator_memory()
            .unwrap()
            .active_bytes(),
        before
    );
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run explicitly with an accessible Metal device"
)]
fn native_embedded_startup_forecasts_preserve_execution_and_bound_peak() {
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
    for mixed in [false, true] {
        let root = attention_prediction_fixture(mixed);
        let mut prior_upper = None;
        for lookahead in [false, true] {
            let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
                .with_drafting(DraftingPlan::Embedded {
                    max_draft_tokens: 2,
                    lookahead,
                    adaptive_lookahead: false,
                });
            check_cold_forecast(&root.0, &execution, lookahead);
            let mut loaded = LoadedModel::load_execution_plan(
                &MlxBackendFactory::default(),
                &root.0,
                &execution,
            )
            .unwrap();
            let generation = loaded.speculative_generation_options().unwrap().unwrap();
            let (model, drafting) = loaded.parts_mut();
            let chat = model
                .prepare_chat(ChatTemplateRequest {
                    messages: vec![serde_json::json!({"role":"user","content":"left right"})],
                    add_generation_prompt: true,
                    ..Default::default()
                })
                .unwrap();
            let ids = vec![1, 2, 5, 7, 3, 9, 2, 1];
            let settings = PreparedChatGenerationSettings {
                overrides: GenerationConfigOverrides {
                    max_new_tokens: Some(8),
                    temperature: Some(0.0),
                    ..Default::default()
                },
                seed: 42,
                ..Default::default()
            };
            let options = GenerationForecastOptions {
                budget: MemoryBudget {
                    application_limit_bytes: Some(1 << 30),
                    ..Default::default()
                },
                // Exercise the mechanism envelope itself on this tiny fixture;
                // the usual graph/driver allowance would obscure undercounts.
                calibration: ForecastCalibration {
                    graph_driver_bytes: 0,
                    ..Default::default()
                },
                ..Default::default()
            };
            model.synchronize().unwrap();
            let before = eredu_backend_mlx::allocator_memory()
                .unwrap()
                .active_bytes();
            let forecast = model
                .forecast_speculative_token_ids(
                    &ids,
                    settings,
                    &drafting.as_speculative_draft().unwrap(),
                    generation,
                    &options,
                )
                .unwrap();
            let repeated = model
                .forecast_speculative_token_ids(
                    &ids,
                    settings,
                    &drafting.as_speculative_draft().unwrap(),
                    generation,
                    &options,
                )
                .unwrap();
            assert_eq!(
                eredu_backend_mlx::allocator_memory()
                    .unwrap()
                    .active_bytes(),
                before
            );
            assert_eq!(
                forecast.estimate.fit,
                MemoryFit::LikelyFit,
                "{:?}",
                forecast.estimate
            );
            let peak = forecast.estimate.domains[0]
                .generation_peak
                .upper_bytes
                .unwrap();
            let upper = forecast.estimate.domains[0]
                .additional_generation_peak
                .upper_bytes
                .unwrap();
            assert_eq!(
                peak,
                repeated.estimate.domains[0]
                    .generation_peak
                    .upper_bytes
                    .unwrap()
            );
            assert!(forecast.speculative.as_ref().unwrap().draft.is_none());
            assert!(forecast.speculative.as_ref().unwrap().embedded.is_some());
            if let Some(previous) = prior_upper {
                assert!(peak >= previous);
            }
            prior_upper = Some(peak);
            let serialized = serde_json::to_vec(&forecast).unwrap();
            let restored: GenerationForecast = serde_json::from_slice(&serialized).unwrap();
            let recomputed = restored.with_max_output_tokens(8).unwrap();
            assert_eq!(
                peak,
                recomputed.estimate.domains[0]
                    .generation_peak
                    .upper_bytes
                    .unwrap()
            );
            assert!(
                restored
                    .with_max_output_tokens(16)
                    .unwrap()
                    .estimate
                    .domains[0]
                    .generation_peak
                    .upper_bytes
                    .unwrap()
                    >= peak
            );
            assert_eq!(
                forecast
                    .with_prefill_chunk(1)
                    .unwrap()
                    .request
                    .prefill_chunk_tokens,
                ids.len() as u64
            );
            let mut tight = options.clone();
            tight.budget.application_limit_bytes = Some(1);
            assert_eq!(
                model
                    .forecast_speculative_token_ids(
                        &ids,
                        settings,
                        &drafting.as_speculative_draft().unwrap(),
                        generation,
                        &tight,
                    )
                    .unwrap()
                    .estimate
                    .fit,
                MemoryFit::LikelyShortfall
            );

            macro_rules! request {
                () => {
                    PreparedChatSpeculativeGenerationRequest {
                        input: PreparedChatInput::token_ids(&chat, ids.clone()),
                        drafting: drafting.as_speculative_draft().unwrap(),
                        settings,
                        options: generation,
                        caller_stop_sequences: &[],
                        cancellation: Default::default(),
                        on_event: |_| {},
                    }
                };
            }
            reset_local_allocator_peak().unwrap();
            let baseline = model
                .generate_prepared_text_speculative(request!())
                .unwrap();
            model.synchronize().unwrap();
            let growth = eredu_backend_mlx::allocator_memory()
                .unwrap()
                .peak_bytes()
                .saturating_sub(before);
            assert!(
                growth <= upper,
                "lookahead={lookahead}: growth={growth}, upper={upper}"
            );
            assert_eq!(baseline.token_ids().len(), 8);
            let controlled = model
                .with_controlled_text_speculative(request!(), Default::default(), |session| {
                    while session.step()?.is_some() {}
                    Ok(())
                })
                .unwrap();
            assert_eq!(controlled.token_ids(), baseline.token_ids());
            // Repeating a query after completed isolated lanes does not execute another lane.
            let active = eredu_backend_mlx::allocator_memory()
                .unwrap()
                .active_bytes();
            let after = model
                .forecast_speculative_token_ids(
                    &ids,
                    settings,
                    &drafting.as_speculative_draft().unwrap(),
                    generation,
                    &options,
                )
                .unwrap();
            assert_eq!(after.estimate.fit, MemoryFit::LikelyFit);
            if mixed {
                assert!(
                    model
                        .static_memory()
                        .unwrap()
                        .current_device_parameter_conversion_bytes
                        .value()
                        .copied()
                        .unwrap()
                        > 0,
                    "mixed fixture must exercise retained native conversions"
                );
                let future_conversions = |forecast: &GenerationForecast| {
                    forecast
                        .request
                        .domains
                        .iter()
                        .flat_map(|domain| &domain.executions)
                        .chain(
                            forecast
                                .speculative
                                .as_ref()
                                .and_then(|plan| plan.embedded.as_ref())
                                .map(|prediction| &prediction.execution),
                        )
                        .filter_map(|execution| execution.execution_topology.as_ref())
                        .filter_map(|topology| topology.selected_parameter_promotion_bytes)
                        .sum::<u64>()
                };
                assert!(
                    future_conversions(&after) < future_conversions(&forecast),
                    "warm forecast must credit observed conversions to their actual owners"
                );
            }
            assert_eq!(
                eredu_backend_mlx::allocator_memory()
                    .unwrap()
                    .active_bytes(),
                active
            );
            let repeated_run = model
                .generate_prepared_text_speculative(request!())
                .unwrap();
            assert_eq!(repeated_run.token_ids(), baseline.token_ids());
            let limits = CaptureUsage {
                captures: 2048,
                retained_bytes: 8 << 20,
                host_bytes: 8 << 20,
                encoded_bytes: 8 << 20,
            };
            let capture = model
                .prepare_speculative_capture(
                    settings,
                    CapturePlan {
                        schema_version: CAPTURE_SCHEMA_VERSION,
                        selections: vec![CaptureSelection {
                            id: "logits".into(),
                            path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                            schedule: Default::default(),
                            slices: vec![],
                            transform: CaptureTransform::TopCandidates { count: 64 },
                        }],
                        limits: CaptureLimits {
                            per_step: limits,
                            cumulative: limits,
                            physical_native_bytes: None,
                            on_limit: CaptureLimitPolicy::Fail,
                        },
                    },
                )
                .unwrap();
            let captured_options = ControlledSpeculativeOptions {
                capture: Some(capture),
                ..Default::default()
            };
            let records = |captures: Vec<eredu_core::speculative::SpeculativePredictionCapture>| {
                captures
                    .into_iter()
                    .map(|capture| (capture.role, capture.position, capture.capture.records))
                    .collect::<Vec<_>>()
            };
            let mut continuous_records = Vec::new();
            let continuous = model
                .generate_observed_text_speculative(request!(), captured_options.clone(), |step| {
                    continuous_records.extend(records(step.captures));
                    ControlFlow::Continue(())
                })
                .unwrap();
            let mut controlled_records = Vec::new();
            let captured = model
                .with_controlled_text_speculative(request!(), captured_options, |session| {
                    while let Some(step) = session.step()? {
                        controlled_records.extend(records(step.captures));
                    }
                    Ok(())
                })
                .unwrap();
            assert_eq!(continuous.token_ids(), captured.token_ids());
            assert_eq!(continuous.token_ids(), baseline.token_ids());
            assert!(!continuous_records.is_empty());
            assert_eq!(continuous_records, controlled_records);

            // An advanced ordinary cache cannot be reinterpreted as a fresh ordinary request.
            let config = eredu_core::TextGenerationConfig::new(
                model.resolve_generation_config(settings.overrides).unwrap(),
            );
            for token in model.generate_tokens(ids.clone(), config).unwrap() {
                token.unwrap();
            }
            assert_eq!(
                model
                    .forecast_token_ids(&ids, settings, &options)
                    .unwrap()
                    .estimate
                    .fit,
                MemoryFit::InsufficientInformation
            );
            let isolated = model
                .forecast_speculative_token_ids(
                    &ids,
                    settings,
                    &drafting.as_speculative_draft().unwrap(),
                    generation,
                    &options,
                )
                .unwrap();
            assert_eq!(isolated.estimate.fit, MemoryFit::LikelyFit);
            eprintln!("embedded startup: mixed={mixed}, lookahead={lookahead}, peak={peak}, additional={upper}, measured_growth={growth}");
        }
    }
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run explicitly with an accessible Metal device"
)]
fn native_embedded_startup_forecasts_keep_uncovered_mechanisms_unknown() {
    let device = if cfg!(feature = "metal") {
        LocalDevice::Accelerator(0)
    } else {
        LocalDevice::Cpu
    };
    for root in [
        super::super::v3_components::source_with_prediction(true, true, 2),
        super::pooling::source(true),
    ] {
        let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
            .with_drafting(DraftingPlan::Embedded {
                max_draft_tokens: 2,
                lookahead: false,
                adaptive_lookahead: false,
            });
        let mut loaded =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
                .unwrap();
        let generation = loaded.speculative_generation_options().unwrap().unwrap();
        let (model, drafting) = loaded.parts_mut();
        let options = GenerationForecastOptions {
            budget: MemoryBudget {
                application_limit_bytes: Some(1 << 30),
                ..Default::default()
            },
            ..Default::default()
        };
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(8),
                ..Default::default()
            },
            ..Default::default()
        };
        let before = eredu_backend_mlx::allocator_memory()
            .unwrap()
            .active_bytes();
        let forecast = model
            .forecast_speculative_token_ids(
                &[1, 2, 3, 4],
                settings,
                &drafting.as_speculative_draft().unwrap(),
                generation,
                &options,
            )
            .unwrap();
        assert_eq!(forecast.estimate.fit, MemoryFit::InsufficientInformation);
        assert!(forecast.speculative.as_ref().unwrap().embedded.is_some());
        assert!(forecast
            .estimate
            .domains
            .iter()
            .any(|pool| pool.generation_peak.upper_bytes.is_none()));
        assert!(
            forecast
                .estimate
                .uncertainties
                .iter()
                .any(|reason| reason.contains("prediction")),
            "{:?}",
            forecast.estimate.uncertainties
        );
        assert_eq!(
            eredu_backend_mlx::allocator_memory()
                .unwrap()
                .active_bytes(),
            before
        );
    }
}

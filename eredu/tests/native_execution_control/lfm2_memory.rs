use super::*;
use eredu_core::residency::ParameterConversionRetentionPolicy;
use eredu_core::{TextGenerationConfig, WeightTransformationPlan};

// Explicit knobs shared by the memory and bit-exact parity harnesses. The
// default still exercises managed policy; historical unlimited runs opt in.
fn retention_policy() -> Option<ParameterConversionRetentionPolicy> {
    std::env::var("EREDU_LFM2_RETENTION")
        .ok()
        .map(|value| match value.as_str() {
            "disabled" => ParameterConversionRetentionPolicy::Disabled,
            "unlimited" => ParameterConversionRetentionPolicy::Unlimited,
            bytes => ParameterConversionRetentionPolicy::Bounded {
                max_bytes: bytes
                    .parse()
                    .expect("retention is disabled, unlimited, or payload bytes"),
            },
        })
}

fn allocator_limit() -> usize {
    std::env::var("EREDU_LFM2_ALLOCATOR_CACHE_BYTES")
        .map(|value| value.parse().expect("allocator cache payload bytes"))
        .unwrap_or(0)
}

fn assert_retention_bound(model: &LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>) {
    let observation = model.parameter_conversion_retention().unwrap();
    let reports = observation.value().unwrap();
    assert_eq!(
        reports.len(),
        1,
        "one selected loaded execution has one budget"
    );
    let report = &reports[0];
    let usage = report.usage.value().unwrap();
    assert_eq!(
        usage.reserved_payload_bytes, 0,
        "settled publication releases reservations"
    );
    match report.policy.value().unwrap().effective {
        ParameterConversionRetentionPolicy::Disabled => assert_eq!(usage.retained_payload_bytes, 0),
        ParameterConversionRetentionPolicy::Bounded { max_bytes } => {
            assert!(usage.retained_payload_bytes + usage.reserved_payload_bytes <= max_bytes);
        }
        ParameterConversionRetentionPolicy::Unlimited => {}
    }
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run explicitly with an accessible Metal device"
)]
fn native_lfm2_workspace_forecasts_cover_cold_loaded_and_continued_execution() {
    let root = components::lfm2_fixture();
    let path = std::env::var_os("EREDU_LFM2_MEMORY_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.0.clone());
    let lengths: Vec<usize> = std::env::var("EREDU_LFM2_MEMORY_LENGTHS")
        .map(|s| s.split(',').map(|s| s.parse().unwrap()).collect())
        .unwrap_or(vec![1, 17, 39]);
    let quantized = std::env::var_os("EREDU_LFM2_MEMORY_QUANTIZED").is_some();
    let device = if cfg!(feature = "metal") {
        LocalDevice::Accelerator(0)
    } else {
        LocalDevice::Cpu
    };
    let mut plan = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
        .with_parameter_conversion_retention(retention_policy());
    if quantized {
        plan = plan.with_weight_transformation(WeightTransformationPlan::Affine {
            bits: 4,
            group_size: 64,
        });
    }
    let factory = MlxBackendFactory::default();
    let previous = set_local_allocator_cache_limit(allocator_limit()).unwrap();
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            set_local_allocator_cache_limit(self.0).unwrap();
        }
    }
    let _restore = Restore(previous);
    let inspection = eredu_backend_mlx::native::inspect_model_preparation(
        &path,
        eredu_backend_mlx::native::MlxInspectionOptions::new(
            factory.load_request_for_plan(&plan).unwrap(),
        ),
    )
    .unwrap();
    let before_load_active = eredu_backend_mlx::allocator_memory()
        .unwrap()
        .active_bytes();
    let (mut model, _) = LoadedModel::load_execution_plan(&factory, &path, &plan)
        .unwrap()
        .into_parts();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(8),
            temperature: Some(0.0),
            ..Default::default()
        },
        prefill: eredu_core::PrefillChunkPolicy::Bounded(std::num::NonZeroUsize::new(4).unwrap()),
        ..Default::default()
    };
    let options = GenerationForecastOptions {
        budget: MemoryBudget {
            application_limit_bytes: Some(256 << 30),
            ..Default::default()
        },
        ..Default::default()
    };
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let mut measurements = Vec::new();
    for (iteration, length) in lengths.into_iter().enumerate() {
        model.reset().unwrap();
        model.synchronize().unwrap();
        let before = model.static_memory().unwrap();
        let converted_before = *before
            .current_device_parameter_conversion_bytes
            .value()
            .unwrap();
        let ids = vec![1; length];
        let baseline = eredu_backend_mlx::allocator_memory()
            .unwrap()
            .active_bytes();
        let mut cold_options = GenerationMemoryOptions::for_local_device(
            eredu_core::InputTokenCount::text(length as u64),
            plan.device(),
        )
        .unwrap();
        cold_options.max_output_tokens = Some(8);
        cold_options.prefill_chunk_tokens = 4;
        cold_options.budget = options.budget.clone();
        let cold =
            forecast_inspected_generation(&inspection, &cold_options, &Default::default()).unwrap();
        let loaded = model.forecast_token_ids(&ids, settings, &options).unwrap();
        // Recompute the same request without upstream completion knowledge;
        // this isolates lifetime accounting from geometry and native execution.
        let mut lazy_request = loaded.request.clone();
        for domain in &mut lazy_request.domains {
            for execution in &mut domain.executions {
                if let Some(full) = execution
                    .input_score_attention_mechanism
                    .as_mut()
                    .and_then(|facts| facts.full_key_tiles.as_mut())
                {
                    full.evaluates_input_dependencies = false;
                }
            }
        }
        let lazy_upper = estimate_generation_memory(&lazy_request).unwrap().domains[0]
            .additional_generation_peak
            .upper_bytes
            .unwrap();
        assert!(
            loaded.estimate.domains[0]
                .additional_generation_peak
                .upper_bytes
                .unwrap()
                <= lazy_upper
        );
        let mut limited = loaded.request.clone();
        limited.domains[0].budget.application_limit_bytes = Some(16 << 30);
        let budget_16_gib_fit = estimate_generation_memory(&limited).unwrap().fit;
        // Large prompts can still exceed a budget after recalibration. Test
        // shortfall against an actual lower bound instead of an obsolete peak.
        limited.domains[0].budget.application_limit_bytes = Some(
            loaded.estimate.domains[0]
                .generation_peak
                .lower_bytes
                .saturating_sub(1),
        );
        assert_eq!(
            estimate_generation_memory(&limited).unwrap().fit,
            MemoryFit::LikelyShortfall
        );
        assert!(cold.estimate.domains[0].overall_peak.upper_bytes.is_some());
        assert!(loaded.estimate.domains[0]
            .generation_peak
            .upper_bytes
            .is_some());
        if length < 2000 {
            assert_eq!(cold.estimate.fit, MemoryFit::LikelyFit);
            assert_eq!(loaded.estimate.fit, MemoryFit::LikelyFit);
        }
        assert_eq!(cold.request.prefill_chunk_tokens, length as u64);
        assert_eq!(loaded.request.prefill_chunk_tokens, length as u64);
        assert!(loaded.execution.full_pass_reason.is_some());
        assert_eq!(
            loaded.with_prefill_chunk(1).unwrap().estimate.domains[0].generation_peak,
            loaded.estimate.domains[0].generation_peak
        );
        let mut expected_topology = cold.request.domains[0].executions[0]
            .execution_topology
            .clone();
        if let Some(bytes) = expected_topology
            .as_mut()
            .and_then(|w| w.selected_parameter_promotion_bytes.as_mut())
        {
            *bytes = bytes.saturating_sub(converted_before);
        }
        // Phase-6 forecasts credit exact bindings as well as the aggregate.
        // Verify those native observations independently before comparing every
        // remaining architecture/mechanism field with cold inspection.
        let retention = loaded
            .request
            .parameter_conversion_retention
            .as_ref()
            .unwrap();
        assert_eq!(retention.groups.len(), 1);
        let mut credited = 0;
        for conversion in retention.retained_conversions.as_ref().unwrap() {
            let eredu_core::resources::ResourceSize::Fixed { extent } = &conversion.allocation.size
            else {
                panic!("a retained conversion has fixed payload");
            };
            for binding in &conversion.bindings {
                assert_eq!(
                    binding.retention_group.as_ref(),
                    Some(&retention.groups[0].report.group)
                );
                let name = binding.logical_target.as_ref().unwrap();
                let bytes = expected_topology
                    .as_mut()
                    .unwrap()
                    .selected_parameter_promotion_payloads
                    .get_mut(name)
                    .unwrap();
                if *bytes != 0 {
                    assert_eq!(
                        *bytes, extent.payload.lower_bytes,
                        "whole-binding credit for {name}"
                    );
                    credited += *bytes;
                    *bytes = 0;
                }
            }
        }
        assert_eq!(
            credited, converted_before,
            "exact binding credits equal current retained payload"
        );
        assert_eq!(
            expected_topology,
            loaded.request.domains[0].executions[0].execution_topology
        );
        assert!(loaded.request.domains[0].executions[0]
            .execution_topology
            .as_ref()
            .unwrap()
            .layers
            .iter()
            .any(|layer| matches!(
                layer.mixer,
                eredu_runtime::execution_topology::TokenMixerTopology::GatedConvolution { .. }
            )));
        assert_eq!(
            eredu_backend_mlx::allocator_memory()
                .unwrap()
                .active_bytes(),
            baseline
        );
        reset_local_allocator_peak().unwrap();
        let config =
            TextGenerationConfig::new(model.resolve_generation_config(settings.overrides).unwrap());
        let mut sampled_cached_peak = eredu_backend_mlx::allocator_memory()
            .unwrap()
            .cached_bytes();
        let generation_started = std::time::Instant::now();
        let mut tokens = model.generate_tokens(ids.clone(), config).unwrap();
        let mut generated = Vec::new();
        let mut first_token_elapsed = std::time::Duration::ZERO;
        for _ in 0..4 {
            generated.push(tokens.next().unwrap().unwrap().token_id().unwrap());
            sampled_cached_peak = sampled_cached_peak.max(
                eredu_backend_mlx::allocator_memory()
                    .unwrap()
                    .cached_bytes(),
            );
            if generated.len() == 1 {
                first_token_elapsed = generation_started.elapsed();
            }
        }
        tokens.synchronize().unwrap();
        let first_four_elapsed = generation_started.elapsed();
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
        let first_peak = eredu_backend_mlx::allocator_memory().unwrap().peak_bytes();
        reset_local_allocator_peak().unwrap();
        let remainder_started = std::time::Instant::now();
        for token in tokens {
            generated.push(token.unwrap().token_id().unwrap());
            sampled_cached_peak = sampled_cached_peak.max(
                eredu_backend_mlx::allocator_memory()
                    .unwrap()
                    .cached_bytes(),
            );
        }
        model.synchronize().unwrap();
        let remainder_elapsed = remainder_started.elapsed();
        let generation_elapsed = first_four_elapsed + remainder_elapsed;
        let measured = eredu_backend_mlx::allocator_memory().unwrap();
        let measured_growth = first_peak
            .max(measured.peak_bytes())
            .saturating_sub(baseline);
        let continued_growth = measured.peak_bytes().saturating_sub(continuation_baseline);
        let upper = loaded.estimate.domains[0]
            .additional_generation_peak
            .upper_bytes
            .unwrap();
        let continued_upper = continued.estimate.domains[0]
            .additional_generation_peak
            .upper_bytes
            .unwrap();
        assert!(
            measured_growth <= upper,
            "growth {measured_growth}, bound {upper}"
        );
        assert!(continued_growth <= continued_upper);
        assert!(
            first_peak.max(measured.peak_bytes())
                <= loaded.estimate.domains[0]
                    .generation_peak
                    .upper_bytes
                    .unwrap(),
            "native active peak exceeds total modeled generation peak"
        );
        assert_retention_bound(&model);
        let after = model.static_memory().unwrap();
        let converted_after = *after
            .current_device_parameter_conversion_bytes
            .value()
            .unwrap();
        if std::env::var_os("EREDU_LFM2_MEMORY_EXPECT_PARAMETER_CONVERSIONS").is_some() {
            assert!(
                converted_after > 0,
                "mixed-width resident conversions must be reported"
            );
            if converted_before > 0 {
                assert_eq!(
                    converted_after, converted_before,
                    "warm generation must reuse conversions"
                );
            }
        }
        assert_eq!(
            before.logical_parameter_bytes,
            after.logical_parameter_bytes
        );
        assert_eq!(
            after.current_device_resident_bytes.value().unwrap()
                - before.current_device_resident_bytes.value().unwrap(),
            converted_after - converted_before,
        );
        model.reset().unwrap();
        assert_eq!(
            model
                .static_memory()
                .unwrap()
                .current_device_parameter_conversion_bytes,
            after.current_device_parameter_conversion_bytes
        );
        let warm = model.forecast_token_ids(&ids, settings, &options).unwrap();
        let trace = TraceLimits {
            per_record_bytes: 16384,
            total_bytes: 1 << 20,
        };
        let prepared = model
            .prepare_observed_token_ids(&chat, ids, settings, CapturePlan::none(), trace)
            .unwrap();
        let mut ordinary = model
            .forecast_observed_generation(&prepared, &options)
            .unwrap();
        // Available capacity is a fresh host observation on each forecast.
        ordinary.request.domains[0].budget.available_bytes =
            warm.request.domains[0].budget.available_bytes;
        assert_eq!(ordinary.request, warm.request);
        let mut run = model
            .start_controlled_text(prepared, &[], Default::default(), |_| {
                ControlFlow::Continue(())
            })
            .unwrap();
        // Replay the raw iterator's committed tokens. This also prevents fixture
        // semantic EOS differences from making the forecast boundary ambiguous.
        run.force_next_token(generated[0]).unwrap();
        run.step(|_| ControlFlow::Continue(())).unwrap();
        let mut checked_controlled = false;
        if run.finish_reason().is_none() {
            checked_controlled = true;
            let controlled = run.forecast_remaining_generation(4, &options).unwrap();
            assert_eq!(controlled.estimate.fit, MemoryFit::LikelyFit);
            assert_eq!(
                controlled.request.domains[0].executions[0].workspace,
                continued.request.domains[0].executions[0].workspace
            );
            run.run(|_| ControlFlow::Continue(())).unwrap();
        }
        assert_eq!(run.token_ids(), &generated[..run.token_ids().len()]);
        drop(run);
        let usage = model.parameter_conversion_retention().unwrap();
        let mut measurement = serde_json::json!({
            "positions": length, "iteration": iteration,
            "allocator_cache_limit": allocator_limit(), "retention": usage,
            "first_token_ms": first_token_elapsed.as_secs_f64() * 1000.0,
            "generation_ms": generation_elapsed.as_secs_f64() * 1000.0,
            "cached_decode_tokens_per_second": 4.0 / remainder_elapsed.as_secs_f64(),
            "baseline_active": baseline, "active_peak": first_peak.max(measured.peak_bytes()),
            "sampled_cached_peak": sampled_cached_peak,
            "cold_overall_upper": cold.estimate.domains[0].overall_peak.upper_bytes,
            "loaded_generation_upper": loaded.estimate.domains[0].generation_peak.upper_bytes,
            "loaded_estimate": loaded.estimate.domains[0],
            "loaded_additional_upper": upper, "without_evaluation_facts_upper": lazy_upper,
            "measured_growth": measured_growth,
            "continuation_upper": continued_upper, "continuation_growth": continued_growth,
            "tokens": generated,
        });
        if std::env::var_os("EREDU_LFM2_MEMORY_TRIM").is_some() && iteration % 2 == 1 {
            let before_trim = model.parameter_conversion_retention().unwrap();
            let started = std::time::Instant::now();
            let trimmed = model.trim_parameter_conversions().unwrap();
            let trim_ms = started.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(
                trimmed[0].released_payload_bytes,
                before_trim.value().unwrap()[0]
                    .usage
                    .value()
                    .unwrap()
                    .retained_payload_bytes
            );
            assert_eq!(trimmed[0].remaining.retained_payload_bytes, 0);
            assert_eq!(
                model.trim_parameter_conversions().unwrap()[0].released_payload_bytes,
                0
            );
            assert_retention_bound(&model);
            model.reset().unwrap();
            let after_trim = model
                .forecast_token_ids(&vec![1; length], settings, &options)
                .unwrap();
            assert!(
                after_trim.estimate.domains[0]
                    .additional_generation_peak
                    .upper_bytes
                    .unwrap()
                    >= warm.estimate.domains[0]
                        .additional_generation_peak
                        .upper_bytes
                        .unwrap()
            );
            measurement["trim_ms"] = trim_ms.into();
            measurement["trim"] = serde_json::to_value(trimmed).unwrap();
            measurement["post_trim_additional_upper"] = serde_json::to_value(
                after_trim.estimate.domains[0]
                    .additional_generation_peak
                    .upper_bytes,
            )
            .unwrap();
        }
        measurements.push(measurement);

        eprintln!("LFM2 parameter conversions: positions={length}, resident_before={converted_before}, resident_after={converted_after}, warm_additional_upper={}", warm.estimate.domains[0].additional_generation_peak.upper_bytes.unwrap());
        eprintln!("LFM2 memory: model={}, quantized={quantized}, positions={length}, scalar_bytes={}, cold_upper={}, loaded_additional_upper={upper}, measured_growth={measured_growth}, continuation_upper={continued_upper}, continuation_growth={continued_growth}, controlled_forecast={checked_controlled}", path.display(), loaded.request.scalar_bytes, cold.estimate.domains[0].overall_peak.upper_bytes.unwrap());
        eprintln!("LFM2 lifetimes: positions={length}, without_evaluation_facts_upper={lazy_upper}, with_evaluation_facts_upper={upper}");
        eprintln!("LFM2 timing: positions={length}, first_token_ms={:.3}, generation_ms={:.3}, generated_tokens={}", first_token_elapsed.as_secs_f64() * 1000.0, generation_elapsed.as_secs_f64() * 1000.0, generated.len());
        eprintln!(
            "LFM2 recalibration: positions={length}, budget_16_gib_fit={budget_16_gib_fit:?}"
        );
    }
    model.synchronize().unwrap();
    drop(model);
    let immediate_drop_active = eredu_backend_mlx::allocator_memory()
        .unwrap()
        .active_bytes();
    // Session Drop deliberately stages potentially blocking destructors. Exercise
    // the public ordinary-host reclamation boundary, including nested retirements.
    for _ in 0..16 {
        eredu_backend_mlx::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        if eredu_backend_mlx::allocator_memory()
            .unwrap()
            .active_bytes()
            <= before_load_active
        {
            break;
        }
    }
    let after_drop = eredu_backend_mlx::allocator_memory().unwrap();
    assert!(after_drop.active_bytes() <= before_load_active,
        "model drop must release live parameters and conversions: before={before_load_active}, after={}", after_drop.active_bytes());
    if let Some(output) = std::env::var_os("EREDU_LFM2_MEMORY_WRITE") {
        std::fs::write(output, serde_json::to_vec_pretty(&serde_json::json!({
            "measurements": measurements, "before_load_active": before_load_active,
            "immediate_drop_active": immediate_drop_active,
            "after_drop_active": after_drop.active_bytes(), "after_drop_cached": after_drop.cached_bytes(),
        })).unwrap()).unwrap();
    }
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run explicitly with an accessible Metal device"
)]
fn native_lfm2_forecast_recalibration_preserves_logits_and_cached_generation() {
    let root = components::lfm2_fixture();
    let path = std::env::var_os("EREDU_LFM2_MEMORY_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.0.clone());
    let lengths: Vec<usize> = std::env::var("EREDU_LFM2_MEMORY_LENGTHS")
        .map(|s| s.split(',').map(|s| s.parse().unwrap()).collect())
        .unwrap_or(vec![17, 513]);
    let quantized = std::env::var_os("EREDU_LFM2_MEMORY_QUANTIZED").is_some();
    let device = if cfg!(feature = "metal") {
        LocalDevice::Accelerator(0)
    } else {
        LocalDevice::Cpu
    };
    let mut plan = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
        .with_parameter_conversion_retention(retention_policy());
    if quantized {
        plan = plan.with_weight_transformation(WeightTransformationPlan::Affine {
            bits: 4,
            group_size: 64,
        });
    }
    let previous = set_local_allocator_cache_limit(allocator_limit()).unwrap();
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            set_local_allocator_cache_limit(self.0).unwrap();
        }
    }
    let _restore = Restore(previous);
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &path, &plan)
            .unwrap()
            .into_parts();
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(8),
            temperature: Some(0.0),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut results = Vec::new();
    for length in lengths {
        let ids = vec![1; length];
        model.reset().unwrap();
        let config =
            TextGenerationConfig::new(model.resolve_generation_config(settings.overrides).unwrap());
        let raw: Vec<_> = model
            .generate_tokens(ids.clone(), config)
            .unwrap()
            .map(|token| token.unwrap().token_id().unwrap())
            .collect();
        assert_eq!(
            raw.len(),
            8,
            "exercise prefill and seven cached predictions"
        );
        let usage = CaptureUsage {
            captures: 2,
            // Admission also reserves the backing full-prompt logits tensor.
            retained_bytes: 4 << 30,
            host_bytes: 8 << 20,
            encoded_bytes: 8 << 20,
        };
        let capture = CapturePlan {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selections: vec![
                CaptureSelection {
                    id: "prefill-logits".into(),
                    path: "model.logits".into(),
                    schedule: CaptureSchedule {
                        decode: false,
                        ..Default::default()
                    },
                    slices: vec![CaptureSlice {
                        axis: "sequence".into(),
                        start: length as u64 - 1,
                        end: length as u64,
                        stride: 1,
                    }],
                    transform: CaptureTransform::Slice,
                },
                CaptureSelection {
                    id: "decode-logits".into(),
                    path: "model.logits".into(),
                    schedule: CaptureSchedule {
                        prefill: false,
                        ..Default::default()
                    },
                    slices: vec![CaptureSlice {
                        axis: "sequence".into(),
                        start: 0,
                        end: 1,
                        stride: 1,
                    }],
                    transform: CaptureTransform::Slice,
                },
            ],
            limits: CaptureLimits {
                per_step: usage,
                cumulative: CaptureUsage {
                    captures: 16,
                    retained_bytes: 32 << 30,
                    host_bytes: 64 << 20,
                    encoded_bytes: 64 << 20,
                },
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        };
        let trace = TraceLimits {
            per_record_bytes: 8 << 20,
            total_bytes: 64 << 20,
        };
        let mut reference = None;
        for controlled in [false, true] {
            model.reset().unwrap();
            let prepared = model
                .prepare_observed_token_ids(&chat, ids.clone(), settings, capture.clone(), trace)
                .unwrap();
            let mut decisions = Vec::new();
            let mut collect = |record: ObservedGenerationRecord| {
                if let ObservedGenerationEvent::Token {
                    token_id,
                    prediction_index,
                    input_range,
                    forced,
                    captures,
                    ..
                } = record.event
                {
                    assert!(!forced);
                    let captures = captures.unwrap();
                    let records: Vec<_> = captures
                        .records
                        .iter()
                        .filter(|r| r.outcome == CaptureOutcome::Captured)
                        .collect();
                    assert_eq!(records.len(), 1);
                    let Some(CapturePayload::Tensor(tensor)) = &records[0].payload else {
                        panic!("expected full-vocabulary logits");
                    };
                    let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
                        panic!("expected float32 host logits");
                    };
                    assert!(!values.is_empty());
                    assert!(values.iter().all(|v| v.is_finite()));
                    // Store exact bits so baseline JSON round trips preserve rounding.
                    decisions.push(serde_json::json!({
                        "token": token_id, "prediction": prediction_index, "input_range": input_range,
                        "logit_bits": values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    }));
                }
                ControlFlow::Continue(())
            };
            if controlled {
                let mut run = model
                    .start_controlled_text(prepared, &[], Default::default(), |r| {
                        collect(r.generation)
                    })
                    .unwrap();
                // Stop at every committed boundary, including cached decode.
                while run.finish_reason().is_none() {
                    if std::env::var_os("EREDU_LFM2_MEMORY_TRIM").is_some() {
                        run.trim_parameter_conversions().unwrap();
                    }
                    run.step(|r| collect(r.generation)).unwrap();
                }
                assert_eq!(run.token_ids(), raw);
            } else {
                model
                    .generate_observed_text(prepared, &[], Default::default(), &mut collect)
                    .unwrap();
            }
            assert_eq!(decisions.len(), raw.len());
            for (i, (decision, token)) in decisions.iter().zip(&raw).enumerate() {
                assert_eq!(decision["token"], *token);
                assert_eq!(decision["prediction"], i);
                assert_eq!(
                    decision["input_range"],
                    if i == 0 {
                        serde_json::json!([0, length])
                    } else {
                        serde_json::json!([length + i - 1, length + i])
                    }
                );
            }
            if let Some(reference) = &reference {
                assert!(
                    &decisions == reference,
                    "ordinary and controlled logits including cached decode"
                );
            } else {
                reference = Some(decisions);
            }
        }
        eprintln!("LFM2 logit parity: positions={length}, quantized={quantized}, predictions={}, controlled=true", raw.len());
        results.push(serde_json::json!({"positions": length, "decisions": reference.unwrap()}));
    }
    let result = serde_json::json!({"quantized": quantized, "results": results});
    if let Some(path) = std::env::var_os("EREDU_LFM2_PARITY_REFERENCE") {
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert!(
            result == reference,
            "exact logits and tokens before/after forecast recalibration"
        );
    }
    if let Some(path) = std::env::var_os("EREDU_LFM2_PARITY_WRITE") {
        std::fs::write(path, serde_json::to_vec(&result).unwrap()).unwrap();
    }
}

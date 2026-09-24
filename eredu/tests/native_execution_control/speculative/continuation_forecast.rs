//! Live embedded outlooks observe settled owners without executing another step.
use super::*;
use eredu_core::generation::SpeculativeRequestStatus;
use eredu_core::residency::{ParameterConversionRetentionPolicy, ParameterConversionTrimError};
use eredu_runtime::memory_forecast::{
    estimate_speculative_continuation_memory, SpeculativeContinuationForecast,
};

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run explicitly with an accessible Metal device"
)]
fn native_embedded_continuation_forecasts_preserve_settled_state_and_bound_growth() {
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
        let fixture = super::startup_forecast::attention_prediction_fixture(mixed);
        for lookahead in [false, true] {
            let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
                .with_required_session_capabilities(SessionCapabilities::new(true, true, true))
                .with_parameter_conversion_retention(Some(
                    ParameterConversionRetentionPolicy::Bounded { max_bytes: 1024 },
                ))
                .with_drafting(DraftingPlan::Embedded {
                    max_draft_tokens: 2,
                    lookahead,
                    adaptive_lookahead: false,
                });
            let mut loaded = LoadedModel::load_execution_plan(
                &MlxBackendFactory::default(),
                &fixture.0,
                &execution,
            )
            .unwrap();
            let generation = loaded.speculative_generation_options().unwrap().unwrap();
            let (model, drafting) = loaded.parts_mut();
            let chat = model
                .prepare_chat(ChatTemplateRequest {
                    messages: vec![serde_json::json!({"role":"user", "content":"left right"})],
                    add_generation_prompt: true,
                    ..Default::default()
                })
                .unwrap();
            let ids = vec![1, 2, 5, 7, 3, 9, 2, 1];
            let prompt = ids.len() as u64;
            let settings = PreparedChatGenerationSettings {
                overrides: GenerationConfigOverrides {
                    max_new_tokens: Some(16),
                    temperature: Some(if lookahead { 0.8 } else { 0.0 }),
                    ..Default::default()
                },
                seed: 42,
                ..Default::default()
            };
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
            let expected = model
                .generate_prepared_text_speculative(request!())
                .unwrap();
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
            let output = model.with_controlled_text_speculative(
                request!(),
                ControlledSpeculativeOptions {
                    trace_limits: TraceLimits { per_record_bytes: 16 << 10, total_bytes: 256 << 10 },
                    snapshots: Some(SnapshotLimits {
                        max_snapshots: 1, max_branches: 1,
                        retained_bytes: 2 << 30, cumulative_copy_bytes: 8 << 30,
                    }),
                    ..Default::default()
                },
                |session| {
                    assert!(session.forecast_remaining_generation(8, &options).is_err());
                    assert!(matches!(session.trim_parameter_conversions(),
                        Err(ParameterConversionTrimError::NotQuiescent)));
                    session.step()?.unwrap();
                    assert_eq!(session.status(), SpeculativeRequestStatus::ReadyToDraft);
                    let tokens = session.token_ids().to_vec();
                    let sampling = session.sampling_state();
                    let epoch = session.epoch();
                    let usage = session.snapshot_usage();
                    let before = eredu_backend_mlx::allocator_memory().unwrap().active_bytes();
                    let remaining = 16 - tokens.len() as u64;
                    let forecast = session.forecast_remaining_generation(remaining, &options).unwrap();
                    let zero = session.forecast_remaining_generation(0, &options).unwrap();
                    let short = session.forecast_remaining_generation(1, &options).unwrap();
                    let repeated = session.forecast_remaining_generation(remaining, &options).unwrap();
                    assert_eq!(forecast.estimate.fit, MemoryFit::LikelyFit, "{:?}", forecast.estimate);
                    assert!(forecast.speculative.embedded.is_some());
                    assert!(forecast.speculative.draft.is_none());
                    assert!(forecast.continuation.draft.is_none());
                    let prediction = forecast.continuation.embedded.as_ref().unwrap();
                    assert_eq!(prediction.layer_positions.len(), 2);
                    assert!(prediction.layer_positions.iter().all(|&n| n > 0 && n < prompt));
                    assert!(prediction.current_state.lower_bytes > 0);
                    assert!(prediction.peak_state.upper_bytes.unwrap() >= prediction.current_state.lower_bytes);
                    assert!(prediction.retained_features.upper_bytes.unwrap() > 0);
                    assert_eq!(zero.continuation.embedded.as_ref().unwrap().current_state, prediction.current_state);
                    assert_eq!(forecast.continuation.target.current_positions, prompt);
                    assert!(forecast.continuation.target.current_state.lower_bytes > 0);
                    assert_eq!(forecast.continuation, repeated.continuation);
                    assert_eq!(zero.continuation.target.current_state, forecast.continuation.target.current_state);
                    assert_eq!(zero.estimate.fit, MemoryFit::LikelyFit);
                    assert!(short.estimate.domains[0].generation_peak.upper_bytes.unwrap()
                        <= forecast.estimate.domains[0].generation_peak.upper_bytes.unwrap());
                    for pool in &forecast.estimate.domains {
                        assert!(pool.phases.iter().all(|phase| !matches!(phase.phase,
                            MemoryPhase::Loading | MemoryPhase::Prefill)));
                    }
                    let roundtrip: SpeculativeContinuationForecast = serde_json::from_str(
                        &serde_json::to_string(&forecast).unwrap()).unwrap();
                    let recomputed = estimate_speculative_continuation_memory(
                        &roundtrip.request, &roundtrip.speculative, &roundtrip.continuation).unwrap();
                    assert_eq!(recomputed.domains, forecast.estimate.domains);
                    assert_eq!(session.token_ids(), tokens);
                    assert_eq!(session.sampling_state(), sampling);
                    assert_eq!(session.epoch(), epoch);
                    assert_eq!(session.snapshot_usage(), usage);
                    assert!(eredu_backend_mlx::allocator_memory().unwrap().active_bytes() <= before);
                    let mut tight = options.clone();
                    tight.budget.application_limit_bytes = Some(1);
                    assert_eq!(session.forecast_remaining_generation(remaining, &tight).unwrap().estimate.fit,
                        MemoryFit::LikelyShortfall);
                    let saved = session.snapshot()?;
                    let branch = session.fork(&saved)?;
                    let retained = session.forecast_remaining_generation(remaining, &options).unwrap();
                    assert!(retained.continuation.retained_snapshots.upper_bytes.unwrap() > 0);
                    session.exchange(&branch)?;
                    let exchanged = session.forecast_remaining_generation(remaining, &options).unwrap();
                    assert_eq!(exchanged.continuation.target.current_positions, prompt);
                    let saved_usage = session.snapshot_usage();
                    let trimmed = session.trim_parameter_conversions().unwrap();
                    assert!(trimmed.external_drafter.is_none());
                    assert_eq!(trimmed.target.len(), 1, "embedded owners share one budget");
                    assert_eq!(trimmed.target[0].remaining.retained_payload_bytes, 0);
                    assert!(trimmed.target[0].released_payload_bytes <= 1024);
                    if mixed {
                        assert!(trimmed.target[0].released_payload_bytes > 0);
                    }
                    assert!(trimmed.target[0].reclaimed_backing_bytes.value().is_none());
                    assert_eq!(session.trim_parameter_conversions().unwrap().target[0].released_claims, 0);
                    assert_eq!(session.snapshot_usage(), saved_usage);
                    assert_eq!(session.token_ids(), tokens);
                    assert_eq!(session.sampling_state(), sampling);
                    let after_trim = session.forecast_remaining_generation(remaining, &options).unwrap();
                    assert_eq!(after_trim.continuation, exchanged.continuation);
                    session.exchange(&branch)?;
                    session.release_branch(&branch)?;
                    let forecast = session.forecast_remaining_generation(remaining, &options).unwrap();
                    let baseline = eredu_backend_mlx::allocator_memory().unwrap().active_bytes();
                    reset_local_allocator_peak().unwrap();
                    let mut rejected_pending = false;
                    let mut advanced_boundary = false;
                    let mut accepted = 0;
                    let mut rejected = 0;
                    while let Some(step) = session.step()? {
                        if let Some(verification) = &step.verification {
                            accepted += verification.dispositions.iter().filter(|d| **d == SpeculativeProposalDisposition::Accepted).count();
                            rejected += verification.dispositions.iter().filter(|d| **d == SpeculativeProposalDisposition::Rejected).count();
                        }
                        let status = session.status();
                        let outlook = session.forecast_remaining_generation(4, &options);
                        if !session.can_snapshot() || status == SpeculativeRequestStatus::Completed {
                            assert!(outlook.is_err());
                            rejected_pending |= status != SpeculativeRequestStatus::Completed;
                            if matches!(status, SpeculativeRequestStatus::ReadyToSubmitVerification
                                | SpeculativeRequestStatus::TargetVerificationInFlight) {
                                assert!(matches!(session.trim_parameter_conversions(),
                                    Err(ParameterConversionTrimError::NotQuiescent)));
                            }
                        } else {
                            let outlook = outlook.unwrap();
                            assert_eq!(outlook.estimate.fit, MemoryFit::LikelyFit);
                            assert!(outlook.continuation.target.current_positions >= prompt);
                            advanced_boundary |= outlook.continuation.target.current_positions > prompt;
                        }
                    }
                    assert!(rejected_pending);
                    assert!(rejected > 0, "fixture must exercise rejection and rollback after trim");
                    if lookahead {
                        assert!(accepted > 0, "lookahead fixture must also accept proposals after trim");
                    }
                    eprintln!("Post-trim verification: lookahead={lookahead}, accepted={accepted}, rejected={rejected}");
                    assert!(advanced_boundary);
                    let growth = eredu_backend_mlx::allocator_memory().unwrap().peak_bytes().saturating_sub(baseline);
                    let upper = forecast.estimate.domains[0].additional_generation_peak.upper_bytes.unwrap();
                    assert!(growth <= upper, "mixed={mixed}, lookahead={lookahead}: growth={growth}, forecast={upper}");
                    let committed = session.token_ids().to_vec();
                    session.restore(&saved)?;
                    let replay_usage = session.snapshot_usage();
                    let replay_sampling = session.sampling_state();
                    session.trim_parameter_conversions().unwrap();
                    assert_eq!(session.snapshot_usage(), replay_usage);
                    assert_eq!(session.sampling_state(), replay_sampling);
                    let restored = session.forecast_remaining_generation(remaining, &options).unwrap();
                    assert_eq!(restored.continuation.target.current_positions, prompt);
                    assert_eq!(restored.continuation.target.current_state.lower_bytes, forecast.continuation.target.current_state.lower_bytes);
                    assert_eq!(restored.continuation.embedded.as_ref().unwrap().layer_positions,
                        forecast.continuation.embedded.as_ref().unwrap().layer_positions);
                    assert_eq!(restored.continuation.embedded.as_ref().unwrap().current_state.lower_bytes,
                        forecast.continuation.embedded.as_ref().unwrap().current_state.lower_bytes);
                    session.release_snapshot(&saved)?;
                    while session.step()?.is_some() {}
                    assert_eq!(session.token_ids(), committed);
                    eprintln!("Embedded continuation: mixed={mixed}, lookahead={lookahead}, horizon={remaining}, upper={upper}, measured_growth={growth}, current={:?}", forecast.continuation);
                    Ok(())
                },
            ).unwrap();
            assert_eq!(output.token_ids(), expected.token_ids());
            assert_eq!(output.finish_reason(), expected.finish_reason());
        }
    }
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run explicitly with an accessible Metal device"
)]
fn native_embedded_continuation_forecasts_keep_uncovered_mechanisms_unknown() {
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
        let chat = model
            .prepare_chat(ChatTemplateRequest {
                messages: vec![serde_json::json!({"role":"user","content":"left right"})],
                add_generation_prompt: true,
                ..Default::default()
            })
            .unwrap();
        let options = GenerationForecastOptions {
            budget: MemoryBudget {
                application_limit_bytes: Some(1 << 30),
                ..Default::default()
            },
            ..Default::default()
        };
        model
            .with_controlled_text_speculative(
                PreparedChatSpeculativeGenerationRequest {
                    input: PreparedChatInput::token_ids(&chat, vec![1, 2, 3, 4]),
                    drafting: drafting.as_speculative_draft().unwrap(),
                    settings: PreparedChatGenerationSettings {
                        overrides: GenerationConfigOverrides {
                            max_new_tokens: Some(8),
                            temperature: Some(0.0),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    options: generation,
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event: |_| {},
                },
                Default::default(),
                |session| {
                    session.step()?.unwrap();
                    let before = eredu_backend_mlx::allocator_memory()
                        .unwrap()
                        .active_bytes();
                    let tokens = session.token_ids().to_vec();
                    let outlook = session.forecast_remaining_generation(4, &options).unwrap();
                    assert!(outlook.speculative.embedded.is_some());
                    let zero = session.forecast_remaining_generation(0, &options).unwrap();
                    assert!(zero
                        .continuation
                        .embedded
                        .as_ref()
                        .unwrap()
                        .current_state
                        .upper_bytes
                        .is_none());
                    assert_eq!(zero.estimate.fit, MemoryFit::InsufficientInformation);
                    assert_eq!(outlook.estimate.fit, MemoryFit::InsufficientInformation);
                    assert!(
                        outlook
                            .estimate
                            .uncertainties
                            .iter()
                            .any(|reason| reason.contains("prediction")),
                        "{:?}",
                        outlook.estimate.uncertainties
                    );
                    assert_eq!(session.token_ids(), tokens);
                    assert!(
                        eredu_backend_mlx::allocator_memory()
                            .unwrap()
                            .active_bytes()
                            <= before
                    );
                    while session.step()?.is_some() {}
                    Ok(())
                },
            )
            .unwrap();
    }
}

use super::*;
use eredu::api::{
    ForecastExecutionContract, GenerationForecastOptions, LogitsWorkspace, MemoryBytes, MemoryFit,
};
use eredu_core::{generation::SpeculativeSchedulerOptions, AvailableMemory};
use eredu_runtime::memory_estimation::WorkspaceGeometry;
use eredu_runtime::memory_forecast::GenerationForecastError;
use eredu_runtime::memory_forecast::{
    GenerationForecastBackend, LoadedMemoryGeometry, LoadedMemoryProfile,
};

impl GenerationForecastBackend for MockBackend {
    fn capture_memory_projection(
        _: &ModelRuntime<Self>,
        capture: &eredu_core::capture::AdmittedCapturePlan,
        intervention: Option<&eredu_core::intervention::AdmittedInterventionPlan>,
        first_prediction: u64,
    ) -> Result<Option<eredu_runtime::capture::CaptureUsageProjection>, GenerationForecastError>
    {
        if intervention.is_some_and(|p| !p.plan().operations.is_empty()) {
            return Ok(None);
        }
        eredu_runtime::capture::project_capture_usage(capture, first_prediction, |shape, _, _| {
            observed_mock::cost(shape).map(Some)
        })
        .map_err(|e| CapabilityError::Observation(e.to_string()).into())
    }

    fn loaded_memory_profile(
        _: &ModelRuntime<Self>,
    ) -> Result<LoadedMemoryProfile, GenerationForecastError> {
        Ok(LoadedMemoryProfile {
            geometry: LoadedMemoryGeometry {
                state_layout: StateMemoryLayout::new(
                    eredu_core::LayerSchedule::new(
                        2,
                        vec![
                            eredu_core::cache::LayerCachePolicy::key_only(
                                eredu_core::AttentionPolicy::Full,
                                1,
                                8
                            )
                            .unwrap();
                            2
                        ],
                    )
                    .unwrap(),
                    vec![0; 2],
                    32,
                    8,
                    EstimationCompleteness::Complete,
                )
                .unwrap(),
                workspace: Some(WorkspaceGeometry {
                    hidden_size: 32,
                    intermediate_size: 64,
                    query_width: 32,
                    key_value_width: 8,
                    query_heads: 4,
                    vocabulary_size: 128,
                    gated_convolution: None,
                    input_score_attention: None,
                    mixed_precision_parameter_bytes: None,
                }),
                scalar_bytes: NonZeroU8::new(2).unwrap(),
                fully_resident: true,
                assumptions: vec![],
            },
            parameters: StaticMemoryReport {
                logical_parameter_bytes: Observed::exact(4096, "fixture"),
                current_host_resident_bytes: Observed::exact(0, "fixture"),
                current_device_resident_bytes: Observed::exact(4096, "fixture"),
                planned_disk_backed_bytes: Observed::exact(0, "fixture"),
                backend_active_allocation_bytes: Observed::exact(
                    999999,
                    "unrelated process allocations",
                ),
                backend_allocator_cache_bytes: Observed::exact(0, "fixture"),
                physical_semantics: PhysicalMemorySemantics::Unified,
                currently_cached_shards: Observed::exact(0, "fixture"),
            },
            available: AvailableMemory {
                physical_memory_bytes: Observed::exact(1 << 30, "fixture"),
                available_memory_bytes: Observed::exact(1 << 29, "fixture"),
                physical_semantics: PhysicalMemorySemantics::Unified,
            },
            allocator_cache_limit: Observed::exact(0, "fixture"),
            host_execution: false,
        })
    }
    fn forecast_execution_contract(
        _: &ModelRuntime<Self>,
        prompt: Option<&Self::Prompt>,
        instrumented: bool,
    ) -> ForecastExecutionContract {
        ForecastExecutionContract {
            full_pass_reason: if instrumented {
                Some("capture full pass".into())
            } else if prompt.is_some() {
                Some("structured prompt full pass".into())
            } else {
                None
            },
            logits: if instrumented {
                LogitsWorkspace::EveryPosition
            } else {
                LogitsWorkspace::FinalPosition
            },
        }
    }
}

impl eredu_runtime::memory_forecast::SpeculativeForecastBackend<MockDrafter> for MockBackend {
    fn speculative_memory_profile(
        runtime: &ModelRuntime<Self>,
        drafting: &eredu_core::SpeculativeDraft<'_, MockDrafter>,
    ) -> Result<
        Option<eredu_runtime::memory_forecast::SpeculativeMemoryProfile>,
        GenerationForecastError,
    > {
        if !matches!(drafting, eredu_core::SpeculativeDraft::External(_)) {
            return Ok(None);
        }
        Ok(Some(
            eredu_runtime::memory_forecast::SpeculativeMemoryProfile {
                draft: Some(Self::loaded_memory_profile(runtime)?),
                auxiliary_bytes_per_position: MemoryBytes::exact(0),
                sampling_bytes_per_vocabulary_entry: MemoryBytes::estimated(
                    0,
                    128,
                    "fixture sampling",
                ),
                proposal_capacity: 4,
                shared_allocator: true,
            },
        ))
    }
}

#[test]
fn speculative_forecast_borrows_the_prepared_request_and_preserves_facts_on_recompute() {
    let (model, chat, settings) = setup();
    let mut drafter = MockDrafter;
    let request = PreparedChatSpeculativeGenerationRequest {
        input: PreparedChatInput::token_ids(&chat, vec![1; 17]),
        drafting: eredu_core::SpeculativeDraft::External(&mut drafter),
        settings,
        options: Default::default(),
        caller_stop_sequences: &[],
        cancellation: Default::default(),
        on_event: |_: SemanticEvent| panic!("forecast must not invoke callbacks"),
    };
    let forecast = model
        .forecast_prepared_speculative_generation(&request, &Default::default())
        .unwrap();
    assert_eq!(forecast.estimate.fit, MemoryFit::LikelyFit);
    assert_eq!(forecast.execution.logits, LogitsWorkspace::EveryPosition);
    assert_eq!(forecast.request.prefill_chunk_tokens, 17);
    assert_eq!(
        forecast
            .speculative
            .as_ref()
            .unwrap()
            .draft
            .as_ref()
            .unwrap()
            .domains[0]
            .already_resident_bytes,
        4096
    );
    let raw = model
        .forecast_speculative_token_ids(
            &[1; 17],
            settings,
            &request.drafting,
            request.options,
            &Default::default(),
        )
        .unwrap();
    assert_eq!(forecast.estimate, raw.estimate);
    assert_eq!(forecast.speculative, raw.speculative);
    assert_eq!(
        forecast.with_prefill_chunk(1).unwrap().estimate,
        forecast.estimate
    );
    let shorter = forecast.with_max_output_tokens(2).unwrap();
    assert_eq!(shorter.request.max_output_tokens, Some(2));
    assert_eq!(
        shorter
            .speculative
            .as_ref()
            .unwrap()
            .draft
            .as_ref()
            .unwrap()
            .max_output_tokens,
        Some(2)
    );
    assert!(
        shorter.estimate.domains[0].generation_peak.upper_bytes
            <= forecast.estimate.domains[0].generation_peak.upper_bytes
    );
    assert_eq!(
        shorter.estimate.domains[0].phases[0].parameters.lower_bytes,
        8192
    );
    let json = serde_json::to_value(&forecast).unwrap();
    assert_eq!(
        json["estimate"]["domains"][0]["phases"][2]["phase"],
        "speculative_verification"
    );
    let decoded: eredu::api::GenerationForecast = serde_json::from_value(json).unwrap();
    assert_eq!(
        decoded.with_max_output_tokens(2).unwrap().estimate,
        shorter.estimate
    );
}

#[test]
fn unsupported_speculative_mechanisms_stay_unknown_and_capacity_is_enforced() {
    let (model, _, settings) = setup();
    let embedded = eredu_core::SpeculativeDraft::<MockDrafter>::Embedded;
    let forecast = model
        .forecast_speculative_token_ids(
            &[1; 17],
            settings,
            &embedded,
            Default::default(),
            &Default::default(),
        )
        .unwrap();
    assert_eq!(forecast.estimate.fit, MemoryFit::InsufficientInformation);
    assert!(forecast.speculative.is_none());
    assert_eq!(
        forecast.with_max_output_tokens(2).unwrap().estimate.fit,
        MemoryFit::InsufficientInformation
    );
    let mut drafter = MockDrafter;
    let external = eredu_core::SpeculativeDraft::External(&mut drafter);
    let options = PreparedChatSpeculativeGenerationOptions {
        max_draft_tokens: NonZeroUsize::new(5).unwrap(),
        ..Default::default()
    };
    assert!(model
        .forecast_speculative_token_ids(&[1; 17], settings, &external, options, &Default::default())
        .is_err());
}

#[test]
fn forecasting_preserves_controlled_and_uninterrupted_speculative_parity() {
    for reject in [false, true] {
        for lookahead in [false, true] {
            let mut outcomes = Vec::new();
            let mut forecasts = Vec::new();
            for controlled in [false, true] {
                let (mut model, chat, mut settings) = setup();
                settings.overrides.max_new_tokens = Some(6);
                let mut drafter = MockDrafter;
                let request = PreparedChatSpeculativeGenerationRequest {
                    input: PreparedChatInput::token_ids(
                        &chat,
                        if reject {
                            vec![CONTROL_REJECTION_PROMPT_TOKEN]
                        } else {
                            vec![3, 4]
                        },
                    ),
                    drafting: eredu_core::SpeculativeDraft::External(&mut drafter),
                    settings,
                    options: PreparedChatSpeculativeGenerationOptions {
                        scheduler: SpeculativeSchedulerOptions::default().with_lookahead(lookahead),
                        ..Default::default()
                    },
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event: |_: SemanticEvent| {},
                };
                let forecast = model
                    .forecast_prepared_speculative_generation(&request, &Default::default())
                    .unwrap();
                assert_eq!(forecast.estimate.fit, MemoryFit::LikelyFit);
                forecasts.push(forecast.estimate);
                // Execute the very request just borrowed by forecasting.
                let output = if controlled {
                    model
                        .with_controlled_text_speculative(request, Default::default(), |session| {
                            while session.step()?.is_some() {}
                            Ok(())
                        })
                        .unwrap()
                } else {
                    model.generate_prepared_text_speculative(request).unwrap()
                };
                outcomes.push(output.token_ids().to_vec());
            }
            assert_eq!(outcomes[0], outcomes[1]);
            assert_eq!(forecasts[0], forecasts[1]);
        }
    }
}

fn setup() -> (
    LoadedModel<MockBackend>,
    PreparedChat,
    PreparedChatGenerationSettings,
) {
    let mut model = unicode_model(None);
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
            ..Default::default()
        })
        .unwrap();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(4),
            ..Default::default()
        },
        prefill: eredu_core::PrefillChunkPolicy::Bounded(NonZeroUsize::new(3).unwrap()),
        ..Default::default()
    };
    (model, chat, settings)
}

#[test]
fn prepared_forecast_derives_contract_and_residency_without_execution() {
    let (model, chat, settings) = setup();
    let request = PreparedChatGenerationRequest {
        input: PreparedChatInput::token_ids(&chat, vec![1; 17]),
        settings,
        caller_stop_sequences: &[],
        cancellation: Default::default(),
        on_event: |_: SemanticEvent| panic!("forecast must not execute callbacks"),
    };
    let forecast = model
        .forecast_prepared_generation(&request, &Default::default())
        .unwrap();
    assert_eq!(forecast.request.input.model_positions, 17);
    assert_eq!(forecast.request.max_output_tokens, Some(4));
    assert_eq!(forecast.request.prefill_chunk_tokens, 3);
    assert_eq!(forecast.execution.logits, LogitsWorkspace::FinalPosition);
    let domain = &forecast.request.domains[0];
    assert_eq!(domain.already_resident_bytes, 4096);
    assert_eq!(domain.loading_peak, MemoryBytes::exact(0));
    assert_eq!(
        domain.executions[0].workspace_overlap.upper_live_copies,
        Some(3)
    );
    assert_eq!(forecast.estimate.fit, MemoryFit::LikelyFit);
    let json = serde_json::to_value(&forecast).unwrap();
    assert_eq!(json["execution"]["logits"], "final_position");
    assert_eq!(json["estimate"]["fit"], "likely_fit");
    assert_eq!(json["estimate"]["domains"][0]["domain"]["kind"], "unified");
    let decoded: eredu::api::GenerationForecast = serde_json::from_value(json).unwrap();
    assert_eq!(decoded.request, forecast.request);
    assert_eq!(decoded.estimate, forecast.estimate);
    assert_eq!(decoded.execution, forecast.execution);
    assert!(forecast.with_prefill_chunk(0).is_err());
    assert!(
        forecast.with_prefill_chunk(1).unwrap().estimate.domains[0]
            .generation_peak
            .upper_bytes
            .unwrap()
            < forecast.estimate.domains[0]
                .generation_peak
                .upper_bytes
                .unwrap()
    );
    let again = model
        .forecast_prepared_generation(&request, &Default::default())
        .unwrap();
    assert_eq!(forecast.request, again.request);
}

#[test]
fn budgets_and_specialized_requests_do_not_hide_uncertainty() {
    let (model, _, mut settings) = setup();
    let mut options = GenerationForecastOptions::default();
    options.budget.application_limit_bytes = Some(1);
    let forecast = model
        .forecast_token_ids(&[1; 17], settings, &options)
        .unwrap();
    assert_eq!(forecast.estimate.fit, MemoryFit::LikelyShortfall);
    options.budget.application_limit_bytes = None;
    settings.prefill = eredu_core::PrefillChunkPolicy::Unchunked;
    let mut forecast = model
        .forecast_token_ids(&[1; 17], settings, &options)
        .unwrap();
    assert_eq!(forecast.execution.logits, LogitsWorkspace::FinalPosition);
    assert_eq!(forecast.request.prefill_chunk_tokens, 17);
    assert_eq!(
        forecast
            .with_prefill_chunk(1)
            .unwrap()
            .request
            .prefill_chunk_tokens,
        1
    );
    eredu::api::mark_speculative_forecast(&mut forecast).unwrap();
    assert_eq!(forecast.execution.logits, LogitsWorkspace::EveryPosition);
    assert_eq!(forecast.estimate.fit, MemoryFit::InsufficientInformation);
    assert!(forecast.request.domains[0].staging.upper_bytes.is_none());
}

#[test]
fn controlled_preparation_forecast_matches_ordinary_and_capture_uses_every_row() {
    use eredu::api::TraceLimits;
    let (model, chat, settings) = setup();
    let limits = TraceLimits {
        per_record_bytes: 65536,
        total_bytes: 1024 * 1024,
    };
    let prepared = model
        .prepare_observed_chat(
            &chat,
            settings,
            eredu_core::capture::CapturePlan::none(),
            limits,
        )
        .unwrap();
    let forecast = model
        .forecast_observed_generation(&prepared, &Default::default())
        .unwrap();
    let ordinary = model
        .forecast_token_ids(prepared.prompt_token_ids(), settings, &Default::default())
        .unwrap();
    assert_eq!(forecast.request, ordinary.request);
    assert_eq!(forecast.execution, ordinary.execution);
    let captured = model
        .prepare_observed_chat(&chat, settings, observed_mock::plan(), limits)
        .unwrap();
    let capture_forecast = model
        .forecast_observed_generation(&captured, &Default::default())
        .unwrap();
    assert_eq!(
        capture_forecast.execution.logits,
        LogitsWorkspace::EveryPosition
    );
    assert!(capture_forecast.execution.full_pass_reason.is_some());
    assert_eq!(
        capture_forecast.request.prefill_chunk_tokens,
        capture_forecast.request.input.model_positions
    );
    assert_eq!(
        capture_forecast
            .with_prefill_chunk(1)
            .unwrap()
            .request
            .prefill_chunk_tokens,
        capture_forecast.request.prefill_chunk_tokens
    );
    assert_eq!(capture_forecast.estimate.fit, MemoryFit::LikelyFit);
    assert_eq!(
        capture_forecast.request.domains[0]
            .retained_input
            .lower_bytes,
        ordinary.request.domains[0].retained_input.lower_bytes
    );
    let (other, _, _) = setup();
    assert!(other
        .forecast_observed_generation(&prepared, &Default::default())
        .is_err());
}

#[test]
fn captures_use_geometry_and_interventions_keep_limits_without_consuming_them() {
    use eredu::api::TraceLimits;
    use eredu_core::capture::CaptureTransform;
    let (model, chat, settings) = setup();
    let limits = TraceLimits {
        per_record_bytes: 65536,
        total_bytes: 1024 * 1024,
    };
    let mut plan = observed_mock::plan();
    plan.selections[0].transform = CaptureTransform::TopCandidates { count: 1 };
    plan.limits.per_step.retained_bytes = 500_000;
    let captured = model
        .prepare_observed_chat(&chat, settings, plan.clone(), limits)
        .unwrap();
    let forecast = model
        .forecast_observed_generation(&captured, &Default::default())
        .unwrap();
    assert_eq!(forecast.estimate.fit, MemoryFit::LikelyFit);
    assert_eq!(forecast.execution.logits, LogitsWorkspace::EveryPosition);
    let peak = forecast.estimate.domains[0]
        .generation_peak
        .upper_bytes
        .unwrap();
    // Raising already sufficient quotas cannot inflate a geometry projection.
    plan.limits.cumulative.retained_bytes += 1234;
    plan.limits.cumulative.host_bytes += 5678;
    let larger = model
        .prepare_observed_chat(&chat, settings, plan.clone(), limits)
        .unwrap();
    let larger = model
        .forecast_observed_generation(&larger, &Default::default())
        .unwrap();
    assert_eq!(
        larger.estimate.domains[0]
            .generation_peak
            .upper_bytes
            .unwrap(),
        peak
    );
    let short = forecast.with_max_output_tokens(1).unwrap();
    assert!(
        short.request.domains[0].retained_input.upper_bytes
            < forecast.request.domains[0].retained_input.upper_bytes
    );
    let restored = short
        .with_max_output_tokens(forecast.request.max_output_tokens.unwrap())
        .unwrap();
    assert_eq!(restored.request, forecast.request);
    let wire = serde_json::to_string(&forecast).unwrap();
    let decoded: eredu::api::GenerationForecast = serde_json::from_str(&wire).unwrap();
    assert_eq!(
        decoded.with_max_output_tokens(1).unwrap().request,
        short.request
    );
    assert!(forecast
        .with_max_output_tokens(100)
        .unwrap()
        .request
        .domains[0]
        .retained_input
        .detail
        .contains("admitted-limit fallback"));
    let intervened = model
        .prepare_intervened_chat(
            &chat,
            settings,
            plan,
            observed_mock::intervention_plan(1.0),
            limits,
        )
        .unwrap();
    let intervened = model
        .forecast_observed_generation(&intervened, &Default::default())
        .unwrap();
    assert_eq!(intervened.estimate.fit, MemoryFit::LikelyFit);
    assert!(
        intervened.estimate.domains[0]
            .generation_peak
            .upper_bytes
            .unwrap()
            > larger.estimate.domains[0]
                .generation_peak
                .upper_bytes
                .unwrap()
    );
    let repeated = model
        .forecast_observed_generation(&captured, &Default::default())
        .unwrap();
    assert_eq!(repeated.request, forecast.request);
    assert_eq!(repeated.estimate, forecast.estimate);
    let options = GenerationForecastOptions {
        backend_overhead: Some(MemoryBytes::unknown("unprojected backend")),
        ..Default::default()
    };
    let unknown = model
        .forecast_observed_generation(&captured, &options)
        .unwrap();
    assert_eq!(unknown.estimate.fit, MemoryFit::InsufficientInformation);
    assert!(unknown.estimate.domains[0]
        .generation_peak
        .upper_bytes
        .is_none());
}

#[test]
fn prepared_backend_contract_and_calibration_overrides_are_retained() {
    let (model, chat, settings) = setup();
    let request = PreparedChatGenerationRequest {
        input: PreparedChatInput::prepared_backend_input(&chat, vec![1; 17]),
        settings,
        caller_stop_sequences: &[],
        cancellation: Default::default(),
        on_event: |_: SemanticEvent| {},
    };
    let mut options = GenerationForecastOptions {
        backend_overhead: Some(MemoryBytes::exact(7)),
        ..Default::default()
    };
    options.calibration.workspace_overlap = Some(eredu::api::WorkspaceOverlap::single_layer());
    let forecast = model
        .forecast_prepared_generation(&request, &options)
        .unwrap();
    assert_eq!(forecast.request.prefill_chunk_tokens, 17);
    assert_eq!(
        forecast.execution.full_pass_reason.as_deref(),
        Some("structured prompt full pass")
    );
    assert_eq!(
        forecast.request.domains[0].backend_overhead,
        MemoryBytes::exact(7)
    );
    assert_eq!(
        forecast.request.domains[0].executions[0]
            .workspace_overlap
            .upper_live_copies,
        Some(1)
    );
}

impl eredu::api::ContinuationForecastBackend for MockBackend {
    fn continuation_memory_profile(
        runtime: &ModelRuntime<Self>,
        additional_input_tokens: u64,
    ) -> Result<
        Option<eredu_runtime::memory_forecast::ContinuationMemoryProfile>,
        GenerationForecastError,
    > {
        runtime
            .session()
            .authority
            .require_idle()
            .map_err(eredu_core::BackendFailure::from_error)?;
        let position = runtime.session().cache_positions;
        Ok(Some(
            eredu_runtime::memory_forecast::ContinuationMemoryProfile {
                loaded: Self::loaded_memory_profile(runtime)?,
                plan: eredu::api::ContinuationMemoryPlan {
                    current_positions: position,
                    additional_input_tokens,
                    current_state: MemoryBytes::estimated(
                        0,
                        4096 + 32 * position,
                        "fixture backing",
                    ),
                    peak_state: MemoryBytes::estimated(
                        0,
                        4096 + 32 * (position + additional_input_tokens),
                        "fixture capacity envelope",
                    ),
                },
            },
        ))
    }
}

#[test]
fn ordinary_continuation_observes_pending_decode_once_without_advancing_or_extending_limit() {
    let (mut model, _, _) = setup();
    let config = eredu_core::TextGenerationConfig::new(
        model
            .resolve_generation_config(GenerationConfigOverrides {
                max_new_tokens: Some(4),
                ..Default::default()
            })
            .unwrap(),
    );
    let mut run = model.generate_tokens(vec![1; 17], config).unwrap();
    assert!(run
        .forecast_remaining_generation(3, &Default::default())
        .is_err());
    assert_eq!(run.next().unwrap().unwrap().token_id().unwrap(), 17);
    // No implicit completion polling or settlement from forecasting.
    assert!(run
        .forecast_remaining_generation(3, &Default::default())
        .is_err());
    run.synchronize().unwrap();
    let forecast = run
        .forecast_remaining_generation(100, &Default::default())
        .unwrap();
    assert_eq!(forecast.continuation.current_positions, 17);
    assert_eq!(forecast.estimate.requested_positions, 117);
    assert_eq!(forecast.estimate.fit, MemoryFit::LikelyFit);
    assert_eq!(forecast.estimate.domains[0].phases.len(), 2);
    assert_eq!(forecast.estimate.domains[0].phases[1].query_positions, 1);
    assert_eq!(forecast.request.domains[0].already_resident_bytes, 4096);
    assert_eq!(
        forecast.estimate.domains[0]
            .additional_generation_peak
            .upper_bytes,
        forecast.estimate.domains[0]
            .overall_peak
            .upper_bytes
            .map(|n| n - 4096)
    );
    let zero = run
        .forecast_remaining_generation(0, &Default::default())
        .unwrap();
    assert_eq!(zero.estimate.requested_positions, 17);
    assert_eq!(zero.continuation.current_state.lower_bytes, 17 * 32);
    assert_eq!(zero.continuation.peak_state.lower_bytes, 17 * 32);
    assert_eq!(
        zero.estimate.domains[0].phases[0].persistent_state,
        zero.continuation.current_state
    );
    assert_eq!(zero.estimate.domains[0].phases.len(), 1);
    assert_eq!(
        zero.estimate.domains[0].phases[0].workspace.upper_bytes,
        Some(0)
    );
    let remaining: Vec<_> = run
        .by_ref()
        .take(3)
        .map(|t| t.unwrap().token_id().unwrap())
        .collect();
    assert_eq!(remaining, [18, 19, 20]);
    assert!(run
        .forecast_remaining_generation(1, &Default::default())
        .is_err());
}

#[test]
fn continuation_estimator_keeps_unknown_growth_and_checks_overflow() {
    use eredu::api::{estimate_continuation_memory, ContinuationMemoryPlan};
    let (model, _, settings) = setup();
    let mut request = model
        .forecast_token_ids(&[1; 17], settings, &Default::default())
        .unwrap()
        .request;
    request.max_output_tokens = Some(3);
    let mut plan = ContinuationMemoryPlan {
        current_positions: 17,
        additional_input_tokens: 3,
        current_state: MemoryBytes::estimated(0, 65536, "retained backing after a longer prefix"),
        peak_state: MemoryBytes::unknown("unsupported native growth"),
    };
    let unknown = estimate_continuation_memory(&request, &plan).unwrap();
    assert_eq!(unknown.fit, MemoryFit::InsufficientInformation);
    assert_eq!(
        unknown.domains[0].phases[0].persistent_state.upper_bytes,
        Some(65536)
    );
    assert!(unknown.domains[0].generation_peak.upper_bytes.is_none());
    plan.peak_state = MemoryBytes::estimated(0, 131072, "interior cache/capacity envelope");
    let bounded = estimate_continuation_memory(&request, &plan).unwrap();
    assert_eq!(bounded.fit, MemoryFit::LikelyFit);
    assert_eq!(
        bounded.domains[0].phases[1].persistent_state.upper_bytes,
        Some(131072)
    );
    assert!(bounded.domains[0].phases[1].workspace.upper_bytes.unwrap() >= 131072);
    request.domains[0].budget.application_limit_bytes = Some(4096);
    assert_eq!(
        estimate_continuation_memory(&request, &plan).unwrap().fit,
        MemoryFit::LikelyShortfall
    );
    plan.current_positions = u64::MAX;
    request.input = eredu_core::InputTokenCount::text(u64::MAX);
    assert!(matches!(
        estimate_continuation_memory(&request, &plan),
        Err(eredu_core::CapabilityError::ArithmeticOverflow { .. })
    ));
}

#[test]
fn continuation_native_envelope_bounds_interior_remainder_state() {
    use eredu::api::{estimate_continuation_memory, ContinuationMemoryPlan};
    use eredu_core::cache::{
        LayerCachePolicy, MutableStateResidency, StateTensorDimension, StateTensorDtype,
        StateTensorPolicy, StateTensorRole,
    };
    let (model, _, settings) = setup();
    let mut request = model
        .forecast_token_ids(&[1; 17], settings, &Default::default())
        .unwrap()
        .request;
    request.max_output_tokens = Some(7); // Endpoint is divisible by eight: remainder payload is zero.
    let remainder = StateTensorPolicy::new(
        StateTensorRole::Recurrent,
        vec![
            StateTensorDimension::Batch,
            StateTensorDimension::PrefixTokensRem(std::num::NonZeroU32::new(8).unwrap()),
        ],
        StateTensorDtype::Float32,
        MutableStateResidency::LayerScopedOffloadable,
    )
    .unwrap();
    request.domains[0].executions[0].state_layout = StateMemoryLayout::new(
        eredu_core::LayerSchedule::new(
            1,
            vec![LayerCachePolicy::fixed_only(vec![remainder]).unwrap()],
        )
        .unwrap(),
        vec![0],
        32,
        8,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let plan = ContinuationMemoryPlan {
        current_positions: 17,
        additional_input_tokens: 7,
        current_state: MemoryBytes::estimated(0, 64, "allocated capacity"),
        peak_state: MemoryBytes::estimated(0, 128, "interior peak including retained capacity"),
    };
    let result = estimate_continuation_memory(&request, &plan).unwrap();
    assert_eq!(result.fit, MemoryFit::LikelyFit);
    assert_eq!(
        result.domains[0].phases[1].persistent_state.upper_bytes,
        Some(128)
    );
    assert_eq!(result.domains[0].phases[0].persistent_state.lower_bytes, 4);
    // Endpoint remainder is zero, but the horizon includes the installed tensor.
    assert_eq!(result.domains[0].phases[1].persistent_state.lower_bytes, 4);
    assert_eq!(result.requested_positions, 24);
    assert!(!result
        .uncertainties
        .iter()
        .any(|s| s.contains("interior peaks of remainder-shaped state unavailable")));
}

#[test]
fn continuation_payload_floors_cover_installed_state_without_capacity_or_optional_tensors() {
    use eredu::api::{estimate_continuation_memory, ContinuationMemoryPlan};
    use eredu_core::cache::{
        LayerCachePolicy, MutableStateResidency, StateTensorDimension as Dim, StateTensorDtype,
        StateTensorPolicy, StateTensorRole,
    };
    use std::num::NonZeroU32;
    let (model, _, settings) = setup();
    let mut request = model
        .forecast_token_ids(&[1; 39], settings, &Default::default())
        .unwrap()
        .request;
    request.max_output_tokens = Some(2);
    let native = ContinuationMemoryPlan {
        current_positions: 39,
        additional_input_tokens: 2,
        current_state: MemoryBytes::estimated(0, 65536, "native installed capacity"),
        peak_state: MemoryBytes::estimated(0, 131072, "native growth envelope"),
    };
    let fixed = StateTensorPolicy::new(
        StateTensorRole::Recurrent,
        vec![Dim::Batch, Dim::Fixed(NonZeroU32::new(16).unwrap())],
        StateTensorDtype::Float32,
        MutableStateResidency::LayerScopedOffloadable,
    )
    .unwrap();
    let dense = request.domains[0].executions[0].state_layout.clone();
    let sliding = StateMemoryLayout::new(
        eredu_core::LayerSchedule::new(
            1,
            vec![LayerCachePolicy::key_only(
                eredu_core::AttentionPolicy::Sliding {
                    window: NonZeroU32::new(8).unwrap(),
                },
                1,
                8,
            )
            .unwrap()],
        )
        .unwrap(),
        vec![0],
        32,
        256,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let mixed = StateMemoryLayout::new(
        eredu_core::LayerSchedule::new(
            3,
            vec![
                LayerCachePolicy::key_only(eredu_core::AttentionPolicy::Full, 1, 8).unwrap(),
                LayerCachePolicy::fixed_only(vec![fixed.clone()]).unwrap(),
                LayerCachePolicy::fixed_only(vec![fixed.optional()]).unwrap(),
            ],
        )
        .unwrap(),
        vec![-3, 0, 0],
        32,
        256,
        EstimationCompleteness::PersistentStateOnly,
    )
    .unwrap();
    for (layout, current, peak) in [
        (dense, 39 * 32, 41 * 32),
        (sliding, 8 * 16, 8 * 16),
        (mixed, 36 * 16 + 64, 38 * 16 + 64),
    ] {
        request.domains[0].executions[0].state_layout = layout;
        let bounded = native.with_logical_state_bounds(&request).unwrap();
        assert_eq!(bounded.current_state.lower_bytes, current);
        assert_eq!(bounded.peak_state.lower_bytes, peak);
        assert_eq!(
            bounded.current_state.upper_bytes,
            native.current_state.upper_bytes
        );
        assert_eq!(
            bounded.peak_state.upper_bytes,
            native.peak_state.upper_bytes
        );
        assert_eq!(
            bounded.with_logical_state_bounds(&request).unwrap(),
            bounded
        );
        let forecast = estimate_continuation_memory(&request, &native).unwrap();
        assert_eq!(
            forecast.domains[0].phases[0].persistent_state,
            bounded.current_state
        );
        assert_eq!(
            forecast.domains[0].phases[1].persistent_state,
            bounded.peak_state
        );
        let mut zero = native.clone();
        zero.additional_input_tokens = 0;
        let mut zero_request = request.clone();
        zero_request.max_output_tokens = Some(0);
        let forecast = estimate_continuation_memory(&zero_request, &zero).unwrap();
        assert_eq!(forecast.domains[0].phases.len(), 1);
        assert_eq!(
            forecast.domains[0].phases[0].persistent_state.lower_bytes,
            current
        );
        assert_eq!(
            forecast.domains[0].phases[0].persistent_state.upper_bytes,
            Some(65536)
        );
        let mut unknown = native.clone();
        unknown.current_state = MemoryBytes::unknown("capacity unavailable");
        unknown.peak_state = MemoryBytes::unknown("growth unavailable");
        let unknown = unknown.with_logical_state_bounds(&request).unwrap();
        assert_eq!(unknown.current_state.lower_bytes, current);
        assert_eq!(
            unknown.current_state.kind,
            eredu_core::ObservationKind::Estimated
        );
        assert_eq!(unknown.peak_state.lower_bytes, peak);
        assert!(unknown.current_state.upper_bytes.is_none());
        assert!(unknown.peak_state.upper_bytes.is_none());
        let mut invalid = native.clone();
        invalid.current_state.upper_bytes = Some(current - 1);
        assert!(invalid.with_logical_state_bounds(&request).is_err());
    }
    // An upper-only architecture declaration cannot establish a floor, but
    // neither does it erase an independent backend lower bound.
    request.domains[0].executions[0].state_layout.completeness =
        EstimationCompleteness::Conservative;
    assert_eq!(native.with_logical_state_bounds(&request).unwrap(), native);
    let mut backend_floor = native;
    backend_floor.current_state.lower_bytes = 7;
    backend_floor.peak_state.lower_bytes = 9;
    assert_eq!(
        backend_floor.with_logical_state_bounds(&request).unwrap(),
        backend_floor
    );
}

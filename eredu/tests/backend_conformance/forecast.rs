use super::*;
use eredu::api::{
    ForecastExecutionContract, GenerationForecastOptions, LogitsWorkspace, MemoryBytes, MemoryFit,
};
use eredu_core::AvailableMemory;
use eredu_runtime::memory_estimation::WorkspaceGeometry;
use eredu_runtime::memory_forecast::GenerationForecastError;
use eredu_runtime::memory_forecast::{
    GenerationForecastBackend, LoadedMemoryGeometry, LoadedMemoryProfile,
};

impl GenerationForecastBackend for MockBackend {
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
    assert_eq!(
        capture_forecast.estimate.fit,
        MemoryFit::InsufficientInformation
    );
    let (other, _, _) = setup();
    assert!(other
        .forecast_observed_generation(&prepared, &Default::default())
        .is_err());
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
    let mut options = GenerationForecastOptions::default();
    options.backend_overhead = Some(MemoryBytes::exact(7));
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

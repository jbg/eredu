use super::*;
use eredu_core::{cache::LayerCachePolicy, AttentionPolicy, EstimationCompleteness, LayerSchedule};

fn request() -> GenerationMemoryRequest {
    let state_layout = StateMemoryLayout::new(
        LayerSchedule::new(
            2,
            vec![LayerCachePolicy::key_only(AttentionPolicy::Full, 1, 8).unwrap(); 2],
        )
        .unwrap(),
        vec![0; 2],
        32,
        8,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    GenerationMemoryRequest {
        parameter_conversion_retention: None,
        input: InputTokenCount::text(17),
        max_output_tokens: Some(4),
        forecast_output_tokens: 128,
        batch_size: 1,
        prefill_chunk_tokens: 8,
        scalar_bytes: NonZeroU8::new(2).unwrap(),
        domains: vec![DomainMemoryPlan {
            domain: MemoryDomain::Unified,
            resident_parameters: MemoryBytes::exact(4096),
            already_resident_bytes: 4096,
            retained_input: MemoryBytes::exact(128),
            staging: MemoryBytes::exact(256),
            backend_overhead: MemoryBytes::estimated(0, 512, "test allowance"),
            loading_peak: MemoryBytes::exact(0),
            executions: vec![ExecutionMemoryPlan {
                execution_topology: None,
                input_score_attention_mechanism: None,
                state_layout,
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
                attention: AttentionWorkspace::Materialized,
                cache_update: CacheUpdateWorkspace::InPlace,
                logits: LogitsWorkspace::FinalPosition,
                workspace_overlap: WorkspaceOverlap::single_layer(),
            }],
            budget: MemoryBudget {
                application_limit_bytes: Some(1_000_000),
                available_bytes: None,
                reserve_bytes: 1024,
            },
        }],
    }
}

#[test]
fn phase_overlap_and_uneven_chunks_are_explicit() {
    let result = estimate_generation_memory(&request()).unwrap();
    let domain = &result.domains[0];
    let phases = &domain.phases;
    assert_eq!((phases[0].positions, phases[0].query_positions), (16, 8));
    assert_eq!((phases[1].positions, phases[1].query_positions), (17, 1));
    assert_eq!((phases[2].positions, phases[2].query_positions), (21, 1));
    // Rounded cache: 2 layers * 8 key scalars * 16 positions * 2 bytes.
    assert_eq!(phases[0].persistent_state.lower_bytes, 512);
    assert_eq!(phases[1].persistent_state.lower_bytes, 768);
    let p = &phases[0];
    assert_eq!(
        p.total.lower_bytes,
        p.parameters.lower_bytes
            + p.persistent_state.lower_bytes
            + p.retained_input.lower_bytes
            + p.workspace.lower_bytes
            + p.staging.lower_bytes
    );
    assert!(phases[0].total.lower_bytes > phases[1].total.lower_bytes);
    assert_eq!(
        domain.generation_peak.upper_bytes,
        phases[..3]
            .iter()
            .map(|p| p.total.upper_bytes.unwrap())
            .max()
    );
    assert_eq!(domain.generation_fit, MemoryFit::LikelyFit);
}

#[test]
fn forecast_exposes_growth_without_claiming_a_lifetime_peak() {
    let mut request = request();
    request.max_output_tokens = None;
    let result = estimate_generation_memory(&request).unwrap();
    assert!(result.is_forecast);
    assert_eq!(result.requested_positions, 145);
    assert_eq!(result.domains[0].state_growth_bytes_per_position, 32);
    request.batch_size = 3;
    let result = estimate_generation_memory(&request).unwrap();
    assert_eq!(result.domains[0].state_growth_bytes_per_position, 96);
}

#[test]
fn sliding_and_recurrent_state_use_the_existing_exact_layout() {
    use eredu_core::cache::{StateTensorDtype, StateTensorPolicy, StateTensorRole};
    use std::num::NonZeroU32;
    let mut request = request();
    let recurrent = StateTensorPolicy::new(
        StateTensorRole::Recurrent,
        vec![
            StateTensorDimension::Batch,
            StateTensorDimension::Fixed(NonZeroU32::new(64).unwrap()),
        ],
        StateTensorDtype::Float32,
        eredu_core::cache::MutableStateResidency::LayerScopedOffloadable,
    )
    .unwrap();
    request.domains[0].executions[0].state_layout = StateMemoryLayout::new(
        LayerSchedule::new(
            2,
            vec![
                LayerCachePolicy::key_only(
                    AttentionPolicy::Sliding {
                        window: NonZeroU32::new(4).unwrap(),
                    },
                    1,
                    8,
                )
                .unwrap(),
                LayerCachePolicy::fixed_only(vec![recurrent]).unwrap(),
            ],
        )
        .unwrap(),
        vec![0; 2],
        32,
        8,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let result = estimate_generation_memory(&request).unwrap();
    assert_eq!(result.domains[0].state_growth_bytes_per_position, 0);
    assert_eq!(
        result.domains[0].phases[0].persistent_state.lower_bytes,
        4 * 8 * 2 + 64 * 4
    );
}

#[test]
fn remainder_state_cannot_claim_an_endpoint_is_the_lifetime_peak() {
    use eredu_core::cache::{StateTensorDtype, StateTensorPolicy, StateTensorRole};
    use std::num::NonZeroU32;
    let mut request = request();
    let remainder = StateTensorPolicy::new(
        StateTensorRole::Recurrent,
        vec![
            StateTensorDimension::Batch,
            StateTensorDimension::PrefixTokensRem(NonZeroU32::new(8).unwrap()),
        ],
        StateTensorDtype::Float32,
        eredu_core::cache::MutableStateResidency::LayerScopedOffloadable,
    )
    .unwrap();
    request.domains[0].executions[0].state_layout = StateMemoryLayout::new(
        LayerSchedule::new(
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
    let result = estimate_generation_memory(&request).unwrap();
    assert_eq!(result.fit, MemoryFit::InsufficientInformation);
    assert!(result
        .uncertainties
        .iter()
        .any(|detail| detail.contains("remainder-shaped")));
}

#[test]
fn unknown_overhead_preserves_payload_and_never_silently_fits() {
    let mut request = request();
    request.domains[0].backend_overhead = MemoryBytes::unknown("native graph overhead unavailable");
    let result = estimate_generation_memory(&request).unwrap();
    assert_eq!(result.fit, MemoryFit::InsufficientInformation);
    assert!(result.domains[0].generation_peak.lower_bytes > 4096);
    assert_eq!(result.domains[0].generation_peak.upper_bytes, None);
    request.domains[0].budget.application_limit_bytes = Some(4095);
    assert_eq!(
        estimate_generation_memory(&request).unwrap().fit,
        MemoryFit::LikelyShortfall
    );
}

#[test]
fn loading_and_generation_fit_are_distinct() {
    let mut request = request();
    request.domains[0].loading_peak = MemoryBytes::unknown("conversion source overlap unknown");
    let result = estimate_generation_memory(&request).unwrap();
    assert_eq!(result.domains[0].generation_fit, MemoryFit::LikelyFit);
    assert_eq!(result.fit, MemoryFit::InsufficientInformation);
    request.domains[0].loading_peak = MemoryBytes::exact(2_000_000);
    let result = estimate_generation_memory(&request).unwrap();
    assert_eq!(result.domains[0].overall_peak.lower_bytes, 2_000_000);
    assert_eq!(result.fit, MemoryFit::LikelyShortfall);
}

#[test]
fn observed_availability_gives_a_verdict_without_an_application_budget() {
    let mut request = request();
    let mut available = AvailableMemory {
        physical_memory_bytes: Observed::exact(2_000_000, "fixture"),
        available_memory_bytes: Observed::Available {
            value: 1_000_000,
            kind: eredu_core::ObservationKind::Estimated,
            source: "point-in-time host observation".into(),
        },
        physical_semantics: eredu_core::PhysicalMemorySemantics::Unified,
    };
    request.domains[0].budget = MemoryBudget::from_available_memory(&available, None, 1024);
    assert_eq!(
        estimate_generation_memory(&request).unwrap().fit,
        MemoryFit::LikelyFit
    );
    available.available_memory_bytes = Observed::exact(0, "exhausted host capacity");
    request.domains[0].budget = MemoryBudget::from_available_memory(&available, None, 1024);
    assert_eq!(
        estimate_generation_memory(&request).unwrap().fit,
        MemoryFit::LikelyShortfall
    );
    available.available_memory_bytes = Observed::unavailable("host query failed");
    request.domains[0].budget = MemoryBudget::from_available_memory(&available, None, 1024);
    assert_eq!(
        estimate_generation_memory(&request).unwrap().fit,
        MemoryFit::InsufficientInformation
    );
}

#[test]
fn total_budget_and_incremental_availability_use_different_baselines() {
    let mut request = request();
    let peak = estimate_generation_memory(&request).unwrap().domains[0]
        .generation_peak
        .upper_bytes
        .unwrap();
    request.domains[0].budget = MemoryBudget {
        application_limit_bytes: Some(peak),
        available_bytes: Some(peak - 4096),
        reserve_bytes: 0,
    };
    assert_eq!(
        estimate_generation_memory(&request).unwrap().fit,
        MemoryFit::LikelyFit
    );
    request.domains[0].budget.reserve_bytes = 513;
    assert_eq!(
        estimate_generation_memory(&request).unwrap().fit,
        MemoryFit::LikelyShortfall
    );
}

#[test]
fn a_separate_device_shortfall_cannot_borrow_another_devices_capacity() {
    let mut request = request();
    request.domains[0].domain = MemoryDomain::Device("0".into());
    let mut second = request.domains[0].clone();
    second.domain = MemoryDomain::Device("1".into());
    second.budget.application_limit_bytes = Some(1);
    request.domains.push(second);
    let result = estimate_generation_memory(&request).unwrap();
    assert_eq!(result.domains[0].fit, MemoryFit::LikelyFit);
    assert_eq!(result.domains[1].fit, MemoryFit::LikelyShortfall);
    assert_eq!(result.fit, MemoryFit::LikelyShortfall);
}

#[test]
fn concurrent_replicas_share_a_pool_but_keep_distinct_state() {
    let mut request = request();
    let before = estimate_generation_memory(&request).unwrap();
    let replica = request.domains[0].executions[0].clone();
    request.domains[0].executions.push(replica);
    let after = estimate_generation_memory(&request).unwrap();
    assert_eq!(
        after.domains[0].phases[0].persistent_state.lower_bytes,
        2 * before.domains[0].phases[0].persistent_state.lower_bytes
    );
    assert_eq!(
        after.domains[0].phases[0].parameters,
        before.domains[0].phases[0].parameters
    );
}

#[test]
fn candidate_estimates_recompute_state_workspace_and_final_chunk() {
    let mut request = request();
    request.input = InputTokenCount::text(4097);
    let candidates = estimate_prefill_candidates(&request, &[4097, 512, 64]).unwrap();
    assert!(
        candidates[0].1.domains[0].generation_peak.lower_bytes
            > candidates[1].1.domains[0].generation_peak.lower_bytes
    );
    assert!(
        candidates[1].1.domains[0].generation_peak.lower_bytes
            > candidates[2].1.domains[0].generation_peak.lower_bytes
    );
    assert_eq!(candidates[2].1.domains[0].phases[1].query_positions, 1);
    let small_peak = candidates[1].1.domains[0].overall_peak.upper_bytes.unwrap();
    request.domains[0].budget.application_limit_bytes = Some(small_peak + 1024);
    let decisions = estimate_prefill_candidates(&request, &[4097, 512]).unwrap();
    assert_eq!(decisions[0].1.fit, MemoryFit::LikelyShortfall);
    assert_eq!(decisions[1].1.fit, MemoryFit::LikelyFit);
}

#[test]
fn fused_attention_and_cache_copy_use_selected_mechanism_facts() {
    let mut request = request();
    let baseline = estimate_generation_memory(&request).unwrap();
    request.domains[0].executions[0].attention = AttentionWorkspace::Fused {
        scratch: MemoryBytes::exact(128),
    };
    request.domains[0].executions[0].cache_update = CacheUpdateWorkspace::CopyState;
    let changed = estimate_generation_memory(&request).unwrap();
    let before = &baseline.domains[0].phases[0];
    let after = &changed.domains[0].phases[0];
    assert_eq!(
        after.workspace.lower_bytes,
        before.workspace.lower_bytes - 4 * 8 * 16 * 8 + 128 + 512
    );
}

#[test]
fn conservative_fused_fallback_does_not_claim_score_matrices_are_allocated() {
    let mut request = request();
    request.input = InputTokenCount::text(2048);
    request.prefill_chunk_tokens = 2048;
    request.domains[0].executions[0].attention = AttentionWorkspace::ScoreMatrixUpperBound;
    let estimate = estimate_generation_memory(&request).unwrap();
    let lower = estimate.domains[0].generation_peak.lower_bytes;
    let upper = estimate.domains[0].generation_peak.upper_bytes.unwrap();
    assert!(upper > lower * 10);
    request.domains[0].budget.application_limit_bytes = Some(lower + 1024);
    assert_eq!(
        estimate_generation_memory(&request).unwrap().fit,
        MemoryFit::InsufficientInformation
    );
    assert!(estimate
        .uncertainties
        .iter()
        .any(|detail| detail.contains("score-matrix")));
}

#[test]
fn selected_graph_overlap_scales_linear_intermediates_only() {
    let mut request = request();
    let baseline = estimate_generation_memory(&request).unwrap();
    request.domains[0].executions[0].workspace_overlap = WorkspaceOverlap {
        upper_live_copies: Some(3),
        detail: "three equivalent layer workspaces from a backend calibration".into(),
    };
    let estimate = estimate_generation_memory(&request).unwrap();
    let before = &baseline.domains[0].phases[0];
    let after = &estimate.domains[0].phases[0];
    assert_eq!(after.workspace.lower_bytes, before.workspace.lower_bytes);
    // Two extra copies of the selected local layer's linear intermediates.
    // Vocabulary rows, score matrices and cache storage are counted separately.
    let extra_linear = 2 * 8 * (4 * 32 + 3 * 64 + 32 + 2 * 8) * 2;
    assert_eq!(
        after.workspace.upper_bytes.unwrap(),
        before.workspace.upper_bytes.unwrap() + extra_linear
    );
    assert_eq!(after.persistent_state, before.persistent_state);
    assert!(estimate
        .uncertainties
        .iter()
        .any(|detail| detail.contains("three equivalent")));
}

#[test]
fn unknown_graph_retention_remains_unknown_and_overlap_math_is_checked() {
    let mut request = request();
    request.domains[0].executions[0].workspace_overlap = WorkspaceOverlap::unknown();
    assert_eq!(
        estimate_generation_memory(&request).unwrap().fit,
        MemoryFit::InsufficientInformation
    );
    request.domains[0].executions[0]
        .workspace_overlap
        .upper_live_copies = Some(0);
    assert!(estimate_generation_memory(&request).is_err());
    request.domains[0].executions[0]
        .workspace_overlap
        .upper_live_copies = Some(u64::MAX);
    assert!(matches!(
        estimate_generation_memory(&request),
        Err(CapabilityError::ArithmeticOverflow { .. })
    ));
}

#[test]
fn loading_includes_explicit_backend_overhead_without_adding_generation_peak() {
    let mut request = request();
    request.domains[0].loading_peak = MemoryBytes::exact(900_000);
    let result = estimate_generation_memory(&request).unwrap();
    assert_eq!(result.domains[0].overall_peak.upper_bytes, Some(900_512));
    let loading = result.domains[0].phases.last().unwrap();
    assert_eq!(loading.total.upper_bytes, Some(900_512));
    assert_eq!(loading.backend_overhead.upper_bytes, Some(512));
}

#[test]
fn a_short_prompt_without_output_has_one_prefill_phase() {
    let mut request = request();
    request.input = InputTokenCount::text(3);
    request.max_output_tokens = Some(0);
    let result = estimate_generation_memory(&request).unwrap();
    assert_eq!(result.domains[0].phases.len(), 2);
    assert_eq!(result.domains[0].phases[0].query_positions, 3);
    assert_eq!(result.domains[0].phases[1].phase, MemoryPhase::Loading);
}

#[test]
fn quantization_auxiliaries_remain_in_resident_payload() {
    let mut request = request();
    let packed_weights = 1024;
    let scales_and_biases = 128;
    request.domains[0].resident_parameters = MemoryBytes::exact(packed_weights + scales_and_biases);
    let result = estimate_generation_memory(&request).unwrap();
    assert_eq!(result.domains[0].phases[0].parameters.lower_bytes, 1152);
}

#[test]
fn missing_geometry_and_missing_limits_remain_uncertain() {
    let mut request = request();
    request.domains[0].executions[0].workspace = None;
    assert_eq!(
        estimate_generation_memory(&request).unwrap().fit,
        MemoryFit::InsufficientInformation
    );
    let mut request = super::tests::request();
    request.domains[0].budget = MemoryBudget::default();
    assert_eq!(
        estimate_generation_memory(&request).unwrap().fit,
        MemoryFit::InsufficientInformation
    );
}

#[test]
fn malformed_intervals_duplicate_domains_and_arithmetic_overflow_are_errors() {
    let mut bad = request();
    bad.domains[0].backend_overhead = MemoryBytes::estimated(2, 1, "invalid");
    assert!(estimate_generation_memory(&bad).is_err());
    let mut bad = request();
    bad.domains.push(bad.domains[0].clone());
    assert!(estimate_generation_memory(&bad).is_err());
    let mut bad = request();
    bad.max_output_tokens = Some(u64::MAX);
    assert!(matches!(
        estimate_generation_memory(&bad),
        Err(CapabilityError::ArithmeticOverflow { .. })
    ));
    let mut bad = request();
    bad.domains[0].executions[0]
        .workspace
        .as_mut()
        .unwrap()
        .intermediate_size = u64::MAX;
    assert!(matches!(
        estimate_generation_memory(&bad),
        Err(CapabilityError::ArithmeticOverflow { .. })
    ));
    let mut bad = request();
    bad.domains[0].budget.reserve_bytes = u64::MAX;
    assert!(matches!(
        estimate_generation_memory(&bad),
        Err(CapabilityError::ArithmeticOverflow { .. })
    ));
}

fn static_report(semantics: PhysicalMemorySemantics) -> StaticMemoryReport {
    StaticMemoryReport {
        parameter_conversion_retention:
            eredu_core::residency::unreported_parameter_conversion_retention(),
        logical_parameter_bytes: Observed::exact(100, "test"),
        current_host_resident_bytes: Observed::exact(100, "test"),
        current_device_resident_bytes: Observed::exact(100, "test"),
        current_device_parameter_conversion_bytes: Observed::exact(0, "no retained conversions"),
        planned_disk_backed_bytes: Observed::exact(0, "test"),
        backend_active_allocation_bytes: Observed::exact(1000, "global, not extra"),
        backend_allocator_cache_bytes: Observed::exact(1000, "global, not extra"),
        physical_semantics: semantics,
        currently_cached_shards: Observed::exact(0, "test"),
    }
}

#[test]
fn unified_capacity_is_not_assumed_to_mean_shared_backing() {
    let report = static_report(PhysicalMemorySemantics::Unified);
    let unknown = static_parameter_placement(&report, None).unwrap();
    assert_eq!(
        (unknown[0].1.lower_bytes, unknown[0].1.upper_bytes),
        (100, Some(200))
    );
    let views = static_parameter_placement(&report, Some(100)).unwrap();
    assert_eq!(
        (views[0].1.lower_bytes, views[0].1.upper_bytes),
        (100, Some(100))
    );
    let copies = static_parameter_placement(&report, Some(0)).unwrap();
    assert_eq!(
        (copies[0].1.lower_bytes, copies[0].1.upper_bytes),
        (200, Some(200))
    );
    assert!(static_parameter_placement(&report, Some(101)).is_err());
    let separate =
        static_parameter_placement(&static_report(PhysicalMemorySemantics::SeparateTiers), None)
            .unwrap();
    assert_eq!(separate.len(), 2);
    assert_eq!(separate[0].1.lower_bytes, 100);
    assert_eq!(separate[1].1.lower_bytes, 100);
}

#[test]
fn forecast_wire_enums_use_snake_case_and_preserve_payloads() {
    use serde::{de::DeserializeOwned, Serialize};
    use serde_json::{json, Value};

    fn round_trip<T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug>(
        value: T,
        expected: Value,
    ) {
        assert_eq!(serde_json::to_value(&value).unwrap(), expected);
        assert_eq!(serde_json::from_value::<T>(expected).unwrap(), value);
    }

    for (value, name) in [
        (MemoryFit::LikelyFit, "likely_fit"),
        (MemoryFit::LikelyShortfall, "likely_shortfall"),
        (
            MemoryFit::InsufficientInformation,
            "insufficient_information",
        ),
    ] {
        round_trip(value, json!(name));
    }
    for (value, name) in [
        (MemoryPhase::Loading, "loading"),
        (MemoryPhase::Prefill, "prefill"),
        (MemoryPhase::Decode, "decode"),
    ] {
        round_trip(value, json!(name));
    }
    for (value, name) in [
        (CacheUpdateWorkspace::InPlace, "in_place"),
        (CacheUpdateWorkspace::CopyState, "copy_state"),
        (CacheUpdateWorkspace::Unknown, "unknown"),
    ] {
        round_trip(value, json!(name));
    }
    round_trip(LogitsWorkspace::FinalPosition, json!("final_position"));
    round_trip(LogitsWorkspace::EveryPosition, json!("every_position"));
    round_trip(MemoryDomain::Host, json!({"kind": "host"}));
    round_trip(MemoryDomain::Unified, json!({"kind": "unified"}));
    round_trip(
        MemoryDomain::Device("CUDA:GPU-α/0".into()),
        json!({"kind": "device", "device": "CUDA:GPU-α/0"}),
    );
    for (value, name) in [
        (AttentionWorkspace::Materialized, "materialized"),
        (
            AttentionWorkspace::ScoreMatrixUpperBound,
            "score_matrix_upper_bound",
        ),
        (AttentionWorkspace::Unknown, "unknown"),
    ] {
        round_trip(value, json!({"kind": name}));
    }
    round_trip(
        AttentionWorkspace::Fused {
            scratch: MemoryBytes::estimated(0, 64, "fixture"),
        },
        json!({"kind": "fused", "scratch": {
            "lower_bytes": 0, "upper_bytes": 64, "kind": "estimated", "detail": "fixture"
        }}),
    );
    assert!(serde_json::from_value::<MemoryDomain>(json!({"kind": "device"})).is_err());
    assert!(serde_json::from_value::<AttentionWorkspace>(json!({"kind": "fused"})).is_err());
}

#[test]
fn request_and_estimate_json_share_the_forecast_wire_contract() {
    use serde_json::json;
    let mut request = request();
    request.domains[0].domain = MemoryDomain::Device("gpu:0".into());
    request.domains[0].executions[0].attention = AttentionWorkspace::ScoreMatrixUpperBound;
    request.domains[0].executions[0].cache_update = CacheUpdateWorkspace::CopyState;
    let encoded = serde_json::to_value(&request).unwrap();
    assert_eq!(
        encoded["domains"][0]["domain"],
        json!({"kind": "device", "device": "gpu:0"})
    );
    let execution = &encoded["domains"][0]["executions"][0];
    assert_eq!(
        execution["attention"],
        json!({"kind": "score_matrix_upper_bound"})
    );
    assert_eq!(execution["cache_update"], "copy_state");
    assert_eq!(execution["logits"], "final_position");
    assert_eq!(
        serde_json::from_value::<GenerationMemoryRequest>(encoded.clone()).unwrap(),
        request
    );

    let report = estimate_generation_memory(&request).unwrap();
    let encoded_report = serde_json::to_value(&report).unwrap();
    assert_eq!(
        encoded_report["domains"][0]["domain"],
        encoded["domains"][0]["domain"]
    );
    assert_eq!(encoded_report["fit"], "likely_fit");
    assert_eq!(encoded_report["domains"][0]["generation_fit"], "likely_fit");
    assert_eq!(
        encoded_report["domains"][0]["phases"][0]["phase"],
        "prefill"
    );
    assert_eq!(encoded_report["domains"][0]["phases"][2]["phase"], "decode");
    assert_eq!(
        encoded_report["domains"][0]["phases"][3]["phase"],
        "loading"
    );
    assert_eq!(
        serde_json::from_value::<GenerationMemoryEstimate>(encoded_report).unwrap(),
        report
    );
}

#[test]
fn gated_convolution_workspace_counts_padding_scratch_and_scheduled_overlap() {
    let mut r = request();
    let execution = &mut r.domains[0].executions[0];
    execution.workspace_overlap.upper_live_copies = Some(3); // two layers plus scratch
    execution.workspace.as_mut().unwrap().gated_convolution = Some(GatedConvolutionWorkspace {
        channels: 16,
        kernel_size: 3,
        layers: 1,
    });
    let execution = execution.clone();
    let with = workspace(&execution, &r, 17, 8, 0).unwrap();
    let mut dense = execution.clone();
    dense.workspace.as_mut().unwrap().gated_convolution = None;
    let without = workspace(&dense, &r, 17, 8, 0).unwrap();
    // One conv layer plus one excess scratch set. Half-precision activation
    // storage still permits float32 convolution scratch/promotion.
    let conv = 16 * (6 * 8 + 2 * (8 + 3 - 1) + 8 * 3 + 3) * 4;
    assert_eq!(
        with.upper_bytes.unwrap() - without.upper_bytes.unwrap(),
        conv * 2 + (4 * 32 + 3 * 64 + 32 + 2 * 8) * 8 * 2 * 3
    );
    assert_eq!(with.lower_bytes, without.lower_bytes);
    let mut decode = dense.clone();
    decode.workspace.as_mut().unwrap().gated_convolution = Some(GatedConvolutionWorkspace {
        channels: 16,
        kernel_size: 1,
        layers: 1,
    });
    let small = workspace(&decode, &r, 17, 1, 0).unwrap();
    let plain = workspace(&dense, &r, 17, 1, 0).unwrap();
    assert_eq!(
        small.upper_bytes.unwrap() - plain.upper_bytes.unwrap(),
        16 * (6 + 2 + 1 + 1) * 4 * 2 + (4 * 32 + 3 * 64 + 32 + 2 * 8) * 2 * 3
    );
    decode.workspace_overlap = WorkspaceOverlap::unknown();
    assert!(workspace(&decode, &r, 17, 1, 0)
        .unwrap()
        .upper_bytes
        .is_none());
    decode.workspace_overlap = WorkspaceOverlap::single_layer();
    for (channels, kernel_size, layers) in [
        (0, 3, 1),
        (16, 0, 1),
        (16, 3, 0),
        (16, 3, 3),
        (u64::MAX, 3, 1),
    ] {
        decode.workspace.as_mut().unwrap().gated_convolution = Some(GatedConvolutionWorkspace {
            channels,
            kernel_size,
            layers,
        });
        assert!(workspace(&decode, &r, 17, 8, 0).is_err());
    }
}

#[test]
fn convolution_only_does_not_require_attention_scratch_and_old_wire_records_remain_valid() {
    let mut r = request();
    let execution = &mut r.domains[0].executions[0];
    let g = execution.workspace.as_mut().unwrap();
    let old = serde_json::to_value(&g).unwrap();
    assert!(old.get("gated_convolution").is_none());
    assert!(old.get("input_score_attention").is_none());
    let restored: WorkspaceGeometry = serde_json::from_value(old).unwrap();
    assert_eq!(*g, restored);
    g.query_width = 0;
    g.key_value_width = 0;
    g.query_heads = 0;
    g.gated_convolution = Some(GatedConvolutionWorkspace {
        channels: 32,
        kernel_size: 3,
        layers: 2,
    });
    execution.attention = AttentionWorkspace::Unknown;
    let result = estimate_generation_memory(&r).unwrap();
    assert!(result.domains[0].generation_peak.upper_bytes.is_some());
    let wire = serde_json::to_string(&r).unwrap();
    assert_eq!(
        serde_json::from_str::<GenerationMemoryRequest>(&wire).unwrap(),
        r
    );
}

#[test]
fn explicit_score_attention_retains_tiled_projection_copies_and_unknown_native_facts() {
    let mut r = request();
    r.scalar_bytes = NonZeroU8::new(4).unwrap();
    let mut e = r.domains[0].executions[0].clone();
    e.workspace_overlap.upper_live_copies = Some(3);
    let plain = e.clone();
    let facts = InputScoreAttentionMechanism {
        score_tile_elements: 16,
        max_query_rows: 4,
        key_value_copies: 4,
        score_bytes: 16,
        full_key_tiles: None,
    };
    e.workspace.as_mut().unwrap().input_score_attention = Some(InputScoreAttentionWorkspace {
        layers: 1,
        mechanism: Some(facts),
    });
    for (positions, query, tiles) in [(8, 1, 1), (8, 2, 1), (8, 3, 2), (8, 5, 3), (32, 5, 5)] {
        let base = workspace(&plain, &r, positions, query, 0).unwrap();
        let projected = workspace(&e, &r, positions, query, 0).unwrap();
        let g = e.workspace.as_ref().unwrap();
        let expanded = g.query_width * positions * 4 * 4 * tiles;
        let scores = g.query_heads * query * positions * 16;
        assert_eq!(
            projected.upper_bytes.unwrap() - base.upper_bytes.unwrap(),
            (expanded + scores) * 2
        );
        assert_eq!(projected.lower_bytes, base.lower_bytes);
    }
    let wire = serde_json::to_string(&e).unwrap();
    assert_eq!(
        serde_json::from_str::<ExecutionMemoryPlan>(&wire).unwrap(),
        e
    );
    e.workspace
        .as_mut()
        .unwrap()
        .input_score_attention
        .as_mut()
        .unwrap()
        .mechanism = None;
    assert!(workspace(&e, &r, 8, 3, 0).unwrap().upper_bytes.is_none());
    let explicit = e
        .workspace
        .as_mut()
        .unwrap()
        .input_score_attention
        .as_mut()
        .unwrap();
    explicit.mechanism = Some(InputScoreAttentionMechanism {
        score_tile_elements: 0,
        ..facts
    });
    assert!(workspace(&e, &r, 8, 3, 0).is_err());
    e.workspace
        .as_mut()
        .unwrap()
        .input_score_attention
        .as_mut()
        .unwrap()
        .mechanism = Some(facts);
    assert!(workspace(&e, &r, u64::MAX, 3, 0).is_err());
}

#[test]
fn full_key_attention_bounds_shared_layouts_live_scores_and_completed_outputs() {
    let mut r = request();
    r.batch_size = 3;
    r.scalar_bytes = NonZeroU8::new(4).unwrap();
    let mut e = r.domains[0].executions[0].clone();
    e.workspace_overlap.upper_live_copies = Some(3);
    let plain = e.clone();
    let full = FullKeyAttentionTiles {
        max_key_positions: 32,
        shared_key_value_copies: 4,
        max_live_query_tiles: 3,
        retained_output_copies: 2,
    };
    let facts = InputScoreAttentionMechanism {
        score_tile_elements: 32,
        max_query_rows: 4,
        key_value_copies: 4,
        score_bytes: 16,
        full_key_tiles: Some(full),
    };
    e.workspace.as_mut().unwrap().input_score_attention = Some(InputScoreAttentionWorkspace {
        layers: 1,
        mechanism: Some(facts),
    });
    // Untiled, exactly one full batch, partial tile in a batch, multiple
    // batches, a short final batch, and either side of the full-key limit.
    for (positions, query, live_rows) in [
        (2, 12, 12),
        (8, 4, 4),
        (8, 11, 11),
        (8, 12, 12),
        (8, 13, 12),
        (8, 25, 12),
        (32, 3, 3),
        (32, 4, 3),
        (33, 4, 4),
    ] {
        let base = workspace(&plain, &r, positions, query, 0).unwrap();
        let projected = workspace(&e, &r, positions, query, 0).unwrap();
        let g = e.workspace.as_ref().unwrap();
        let expected = if positions <= full.max_key_positions {
            let shared = g.query_width * positions * 4 * 4;
            let scores = g.query_heads * live_rows * (positions + 1) * 16;
            let retained = g.query_width * query * 4 * 2;
            shared + scores + retained
        } else {
            // Blockwise and legacy mechanisms retain the old, intentionally
            // conservative allowance; they do not inherit full-key promises.
            g.query_width * positions * 4 * 4 * query + g.query_heads * query * positions * 16
        };
        assert_eq!(
            projected.upper_bytes.unwrap() - base.upper_bytes.unwrap(),
            expected * r.batch_size * 2,
            "positions={positions}, query={query}"
        );
        assert_eq!(projected.lower_bytes, base.lower_bytes);
    }

    // Promotion changes shared K/V and retained outputs, while score scratch
    // already includes conversion widths. Also include promoted projected logits,
    // one parameter cast set and the overlap fixture's extra half-set.
    r.scalar_bytes = NonZeroU8::new(2).unwrap();
    let base = workspace(&e, &r, 8, 13, 0).unwrap();
    e.workspace
        .as_mut()
        .unwrap()
        .mixed_precision_parameter_bytes = Some(1024);
    let projected = workspace(&e, &r, 8, 13, 0).unwrap();
    let g = e.workspace.as_ref().unwrap();
    assert_eq!(
        projected.upper_bytes.unwrap() - base.upper_bytes.unwrap(),
        (g.query_width * 8 * 2 * 4 + g.query_width * 13 * 2 * 2) * r.batch_size * 2
            + g.vocabulary_size * r.batch_size * 2
            + 1024
            + 512
    );

    let wire = serde_json::to_value(facts).unwrap();
    assert_eq!(
        serde_json::from_value::<InputScoreAttentionMechanism>(wire.clone()).unwrap(),
        facts
    );
    let mut old_wire = wire;
    old_wire.as_object_mut().unwrap().remove("full_key_tiles");
    let legacy = serde_json::from_value::<InputScoreAttentionMechanism>(old_wire).unwrap();
    assert!(legacy.full_key_tiles.is_none());
    assert!(serde_json::to_value(legacy)
        .unwrap()
        .get("full_key_tiles")
        .is_none());

    for malformed in [
        FullKeyAttentionTiles {
            max_key_positions: 0,
            ..full
        },
        FullKeyAttentionTiles {
            shared_key_value_copies: 0,
            ..full
        },
        FullKeyAttentionTiles {
            max_live_query_tiles: 0,
            ..full
        },
        FullKeyAttentionTiles {
            retained_output_copies: 0,
            ..full
        },
    ] {
        e.workspace
            .as_mut()
            .unwrap()
            .input_score_attention
            .as_mut()
            .unwrap()
            .mechanism = Some(InputScoreAttentionMechanism {
            full_key_tiles: Some(malformed),
            ..facts
        });
        assert!(workspace(&e, &r, 8, 13, 0).is_err());
    }
    e.workspace
        .as_mut()
        .unwrap()
        .input_score_attention
        .as_mut()
        .unwrap()
        .mechanism = Some(InputScoreAttentionMechanism {
        full_key_tiles: Some(FullKeyAttentionTiles {
            max_live_query_tiles: u64::MAX,
            ..full
        }),
        ..facts
    });
    assert!(workspace(&e, &r, 8, 13, 0).is_ok());
    assert!(workspace(&e, &r, 8, u64::MAX, 0).is_err());
    e.workspace_overlap = WorkspaceOverlap::unknown();
    assert!(workspace(&e, &r, 8, 13, 0).unwrap().upper_bytes.is_none());
}

#[test]
fn mixed_precision_covers_parameter_casts_and_promoted_state_without_double_charging_weights() {
    let r = request();
    let mut e = r.domains[0].executions[0].clone();
    e.workspace_overlap.upper_live_copies = Some(3);
    let plain = workspace(&e, &r, 8, 3, 100).unwrap();
    e.workspace
        .as_mut()
        .unwrap()
        .mixed_precision_parameter_bytes = Some(1000);
    let promoted = workspace(&e, &r, 8, 3, 100).unwrap();
    assert_eq!(promoted.lower_bytes, plain.lower_bytes);
    // F32 activation payload (368 scalars per row, 3 rows, 3 live copies),
    // F32 projected logits (128 scalars) with the existing F32 sampling copy
    // unchanged, one cast set plus half an excess set, and promoted state.
    let widened_activations_and_logits = 368 * 3 * 3 * 2 + 128 * 2;
    assert_eq!(
        promoted.upper_bytes.unwrap() - plain.upper_bytes.unwrap(),
        widened_activations_and_logits + 1000 + 500 + 200
    );
    // All parameter casts can already be resident without undoing activation,
    // logit or state promotion; Some(0) retains that execution-width evidence.
    e.workspace
        .as_mut()
        .unwrap()
        .mixed_precision_parameter_bytes = Some(0);
    let warm = workspace(&e, &r, 8, 3, 100).unwrap();
    assert_eq!(warm.lower_bytes, plain.lower_bytes);
    assert_eq!(
        warm.upper_bytes.unwrap() - plain.upper_bytes.unwrap(),
        widened_activations_and_logits + 200
    );
    assert_eq!(
        promoted.upper_bytes.unwrap() - warm.upper_bytes.unwrap(),
        1500
    );
    e.workspace_overlap = WorkspaceOverlap::unknown();
    assert!(workspace(&e, &r, 8, 3, 100).unwrap().upper_bytes.is_none());
}

fn generic_topology() -> crate::execution_topology::TextExecutionTopology {
    use crate::execution_topology::*;
    let projection = |name: &str, input, output| ProjectionTopology {
        input,
        output,
        format: eredu_checkpoint::LinearFormat::Dense,
        bias: false,
        parameter: name.into(),
    };
    let layer = TextLayerTopology {
        input_projections: Vec::new(),
        mixer: TokenMixerTopology::Attention {
            query_heads: 4,
            kv_heads: 1,
            key_width: 8,
            value_width: 8,
            input_scores: false,
            softcap: false,
            sinks: false,
            projections: vec![
                projection("query", 32, 32),
                projection("key", 32, 8),
                projection("value", 32, 8),
                projection("attention-output", 32, 32),
            ],
            output_gate: false,
            query_key_normalization: false,
            rotary: true,
        },
        feed_forward: FeedForwardTopology::Gated {
            intermediate_size: 64,
            projections: vec![
                projection("gate", 32, 64),
                projection("up", 32, 64),
                projection("down", 64, 32),
            ],
        },
        normalization_count: 2,
    };
    TextExecutionTopology {
        hidden_size: 32,
        vocabulary_size: 128,
        layers: vec![layer.clone(), layer],
        output: projection("output", 32, 128),
        selected_parameter_promotion_bytes: None,
        selected_parameter_promotion_payloads: Default::default(),
        output_invocations: 1,
        output_softcap: false,
        missing: vec![],
    }
}

#[test]
fn generic_default_overlap_counts_invocations_including_stateless_layers() {
    let mut r = request();
    let mut topology = generic_topology();
    topology.layers.resize(12, topology.layers[0].clone());
    r.domains[0].executions[0].execution_topology = Some(topology);
    crate::memory_forecast::ForecastCalibration::default()
        .apply(&mut r)
        .unwrap();
    assert_eq!(
        r.domains[0].executions[0]
            .workspace_overlap
            .upper_live_copies,
        Some(15)
    );
}

#[test]
fn generic_target_workspace_does_not_consume_legacy_family_geometry() {
    let mut r = request();
    let execution = &mut r.domains[0].executions[0];
    execution.execution_topology = Some(generic_topology());
    execution.workspace = None;
    let generic = estimate_generation_memory(&r).unwrap();
    assert_eq!(generic.fit, MemoryFit::LikelyFit);
    assert!(generic.domains[0].phases[0].workspace.lower_bytes > 0);
    assert!(!generic
        .uncertainties
        .iter()
        .any(|detail| detail.contains("decoder workspace geometry unavailable")));
    // Contradictory deprecated aggregates cannot alter the selected module path.
    r.domains[0].executions[0].workspace = request().domains[0].executions[0].workspace.clone();
    r.domains[0].executions[0]
        .workspace
        .as_mut()
        .unwrap()
        .hidden_size = u64::MAX;
    assert_eq!(
        estimate_generation_memory(&r).unwrap().domains,
        generic.domains
    );
}

#[test]
fn generic_target_workspace_retention_and_output_contract_are_independent() {
    let mut r = request();
    r.domains[0].executions[0].execution_topology = Some(generic_topology());
    let one = estimate_generation_memory(&r).unwrap().domains[0].phases[0]
        .workspace
        .clone();
    r.domains[0].executions[0]
        .workspace_overlap
        .upper_live_copies = Some(2);
    let two = estimate_generation_memory(&r).unwrap().domains[0].phases[0]
        .workspace
        .clone();
    assert!(two.upper_bytes > one.upper_bytes);
    // No assertion that all native temporaries coexist: the minimum stays one
    // real logical output rather than summing a layer's independent minima.
    assert_eq!(two.lower_bytes, one.lower_bytes);
    r.domains[0].executions[0].logits = LogitsWorkspace::EveryPosition;
    let every = estimate_generation_memory(&r).unwrap().domains[0].phases[0]
        .workspace
        .clone();
    assert!(every.upper_bytes > two.upper_bytes);
    r.domains[0].executions[0]
        .execution_topology
        .as_mut()
        .unwrap()
        .output_softcap = true;
    let capped = estimate_generation_memory(&r).unwrap().domains[0].phases[0]
        .workspace
        .clone();
    assert!(capped.upper_bytes > every.upper_bytes);
    r.domains[0].executions[0]
        .workspace_overlap
        .upper_live_copies = None;
    assert_eq!(
        estimate_generation_memory(&r).unwrap().domains[0].phases[0]
            .workspace
            .upper_bytes,
        None
    );
}

#[test]
fn generic_target_workspace_preserves_unknown_modules_and_checks_geometry() {
    use crate::execution_topology::FeedForwardTopology;
    let mut r = request();
    let mut topology = generic_topology();
    topology.layers[0].feed_forward = FeedForwardTopology::Unknown {
        reason: "custom recurrent update has no mechanism contract".into(),
    };
    r.domains[0].executions[0].execution_topology = Some(topology.clone());
    assert_eq!(
        estimate_generation_memory(&r).unwrap().domains[0].phases[0]
            .workspace
            .upper_bytes,
        None
    );
    topology.layers[0].normalization_count = u64::MAX;
    r.domains[0].executions[0].execution_topology = Some(topology);
    assert!(matches!(
        estimate_generation_memory(&r),
        Err(CapabilityError::ArithmeticOverflow { .. })
    ));
}

#[test]
fn generic_explicit_attention_uses_native_tile_contract_without_score_matrix_fallback() {
    use crate::execution_topology::TokenMixerTopology;
    let mut r = request();
    let mut topology = generic_topology();
    for layer in &mut topology.layers {
        if let TokenMixerTopology::Attention { input_scores, .. } = &mut layer.mixer {
            *input_scores = true;
        }
    }
    r.domains[0].executions[0].execution_topology = Some(topology);
    let unknown = estimate_generation_memory(&r).unwrap();
    assert_eq!(unknown.domains[0].phases[0].workspace.upper_bytes, None);
    r.domains[0].executions[0].input_score_attention_mechanism =
        Some(InputScoreAttentionMechanism {
            score_tile_elements: 16,
            max_query_rows: 2,
            key_value_copies: 4,
            score_bytes: 20,
            full_key_tiles: Some(FullKeyAttentionTiles {
                max_key_positions: 128,
                shared_key_value_copies: 4,
                max_live_query_tiles: 2,
                retained_output_copies: 2,
            }),
        });
    let bounded = estimate_generation_memory(&r).unwrap();
    assert!(bounded.domains[0].phases[0].workspace.upper_bytes.is_some());
    // Fused scratch policy is irrelevant to an explicitly selected input-score kernel.
    r.domains[0].executions[0].attention = AttentionWorkspace::Unknown;
    assert_eq!(
        estimate_generation_memory(&r).unwrap().domains,
        bounded.domains
    );
    // Softcap selects an explicit full matrix even with Fused arithmetic. It
    // must not inherit InputScores query-tile savings or change the arithmetic.
    for layer in &mut r.domains[0].executions[0]
        .execution_topology
        .as_mut()
        .unwrap()
        .layers
    {
        if let TokenMixerTopology::Attention {
            input_scores,
            softcap,
            ..
        } = &mut layer.mixer
        {
            *input_scores = false;
            *softcap = true;
        }
    }
    let softcap = estimate_generation_memory(&r).unwrap();
    assert!(
        softcap.domains[0].phases[0].workspace.upper_bytes
            > bounded.domains[0].phases[0].workspace.upper_bytes
    );
}

#[test]
fn legacy_serialized_workspace_requests_remain_compatible() {
    let r = request();
    let value = serde_json::to_value(&r).unwrap();
    assert!(value["domains"][0]["executions"][0]
        .get("execution_topology")
        .is_none());
    let restored: GenerationMemoryRequest = serde_json::from_value(value).unwrap();
    assert_eq!(
        estimate_generation_memory(&restored).unwrap(),
        estimate_generation_memory(&r).unwrap()
    );
    let mut r = restored;
    r.domains[0].executions[0].execution_topology = Some(generic_topology());
    let restored: GenerationMemoryRequest =
        serde_json::from_value(serde_json::to_value(&r).unwrap()).unwrap();
    assert_eq!(restored, r);
}

#[test]
fn generic_target_rejects_zero_kernels_and_inconsistent_projection_contracts() {
    use crate::execution_topology::*;
    let mut r = request();
    let mut topology = generic_topology();
    topology.output.output += 1;
    r.domains[0].executions[0].execution_topology = Some(topology);
    assert!(estimate_generation_memory(&r).is_err());
    let mut topology = generic_topology();
    topology.layers[0].mixer = TokenMixerTopology::GatedConvolution {
        channels: 32,
        kernel: 0,
        projections: vec![],
    };
    r.domains[0].executions[0].execution_topology = Some(topology);
    assert!(estimate_generation_memory(&r).is_err());
    let mut topology = generic_topology();
    if let TokenMixerTopology::Attention { query_heads, .. } = &mut topology.layers[0].mixer {
        *query_heads = 0;
    }
    r.domains[0].executions[0].execution_topology = Some(topology);
    assert!(estimate_generation_memory(&r).is_err());
}

#[test]
fn generic_expert_mechanism_bounds_selected_routes_without_a_family_dispatch() {
    use crate::execution_topology::*;
    let projection = |name: &str, input, output| ProjectionTopology {
        input,
        output,
        parameter: name.into(),
        format: eredu_checkpoint::LinearFormat::Dense,
        bias: false,
    };
    let mut r = request();
    let mut topology = generic_topology();
    topology.layers[0].feed_forward = FeedForwardTopology::Routed {
        experts: 8,
        selected: 2,
        intermediate_size: 64,
        router: projection("router", 32, 8),
        projections: vec![
            projection("experts-up", 32, 128),
            projection("experts-down", 64, 32),
        ],
    };
    r.domains[0].executions[0].execution_topology = Some(topology.clone());
    let first = estimate_generation_memory(&r).unwrap();
    assert!(first.domains[0].phases[0].workspace.upper_bytes.is_some());
    if let FeedForwardTopology::Routed { selected, .. } = &mut topology.layers[0].feed_forward {
        *selected = 4;
    }
    r.domains[0].executions[0].execution_topology = Some(topology.clone());
    let more_routes = estimate_generation_memory(&r).unwrap();
    assert!(
        more_routes.domains[0].phases[0].workspace.upper_bytes
            > first.domains[0].phases[0].workspace.upper_bytes
    );
    if let FeedForwardTopology::Routed { selected, .. } = &mut topology.layers[0].feed_forward {
        *selected = 9;
    }
    r.domains[0].executions[0].execution_topology = Some(topology);
    assert!(estimate_generation_memory(&r).is_err());
}

/// Frozen before retiring the aggregate estimator, from revision 86a6215e.
/// Old reports remain readable; re-estimation preserves all numerical fields and
/// fit decisions even though the explanatory assumptions describe the new path.
#[test]
fn archived_aggregate_request_and_report_preserve_wire_and_recomputed_bounds() {
    let wire: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/legacy-generation-memory.json"
    ))
    .unwrap();
    let request: GenerationMemoryRequest = serde_json::from_value(wire["request"].clone()).unwrap();
    let report: GenerationMemoryEstimate = serde_json::from_value(wire["report"].clone()).unwrap();
    assert_eq!(serde_json::to_value(&request).unwrap(), wire["request"]);
    assert_eq!(serde_json::to_value(&report).unwrap(), wire["report"]);
    let mut recomputed = estimate_generation_memory(&request).unwrap();
    recomputed.assumptions = report.assumptions.clone();
    assert_eq!(recomputed, report);
}

#[test]
fn archived_aggregate_mechanism_bounds_remain_exact_after_lifetime_lowering() {
    let wire: serde_json::Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/legacy-generation-memory.json"
    ))
    .unwrap();
    let mut r: GenerationMemoryRequest = serde_json::from_value(wire["request"].clone()).unwrap();
    // Values captured from the old evaluator at revision 86a6215e. These exercise
    // prefill, partial-final-chunk and decode, including the old replacement
    // lower bound and simultaneous fused-fallback plus explicit-score allowance.
    let expected = [
        [(10752, Some(10752)), (2048, Some(2048)), (2176, Some(2176))],
        [(11264, Some(11264)), (2816, Some(2816)), (2944, Some(2944))],
        [(11264, Some(52864)), (2816, Some(8800)), (2944, Some(8928))],
        [
            (11264, Some(102016)),
            (2816, Some(19680)),
            (2944, Some(22368)),
        ],
        [
            (11264, Some(76160)),
            (2816, Some(20064)),
            (2944, Some(22752)),
        ],
        [
            (11264, Some(89216)),
            (2816, Some(32352)),
            (2944, Some(37088)),
        ],
        [(11264, None), (2816, None), (2944, None)],
    ];
    for (mode, expected) in expected.into_iter().enumerate() {
        let e = &mut r.domains[0].executions[0];
        match mode {
            1 => e.cache_update = CacheUpdateWorkspace::CopyState,
            2 => {
                e.workspace_overlap.upper_live_copies = Some(3);
                e.workspace.as_mut().unwrap().gated_convolution = Some(GatedConvolutionWorkspace {
                    channels: 16,
                    kernel_size: 3,
                    layers: 1,
                });
            }
            3 => {
                e.workspace.as_mut().unwrap().input_score_attention =
                    Some(InputScoreAttentionWorkspace {
                        layers: 1,
                        mechanism: Some(InputScoreAttentionMechanism {
                            score_tile_elements: 32,
                            max_query_rows: 4,
                            key_value_copies: 4,
                            score_bytes: 16,
                            full_key_tiles: None,
                        }),
                    })
            }
            4 => {
                e.workspace
                    .as_mut()
                    .unwrap()
                    .input_score_attention
                    .as_mut()
                    .unwrap()
                    .mechanism
                    .as_mut()
                    .unwrap()
                    .full_key_tiles = Some(FullKeyAttentionTiles {
                    max_key_positions: 32,
                    shared_key_value_copies: 4,
                    max_live_query_tiles: 3,
                    retained_output_copies: 2,
                })
            }
            5 => {
                e.workspace
                    .as_mut()
                    .unwrap()
                    .mixed_precision_parameter_bytes = Some(1024)
            }
            6 => e.workspace_overlap = WorkspaceOverlap::unknown(),
            _ => (),
        }
        let report = estimate_generation_memory(&r).unwrap();
        let phases = &report.domains[0].phases;
        assert_eq!(phases.len(), 4);
        for (phase, expected) in phases.iter().zip(expected) {
            assert_eq!(
                (phase.workspace.lower_bytes, phase.workspace.upper_bytes),
                expected,
                "legacy mode {mode}: {:?}",
                phase.phase
            );
        }
        assert_eq!(phases[3].workspace, MemoryBytes::exact(0));
        assert_eq!(report.fit == MemoryFit::InsufficientInformation, mode == 6);
    }
}

#[test]
fn zero_query_workspace_has_no_invocation_for_current_or_archived_records() {
    let r = request();
    // An absent invocation cannot project logits, replace installed cache, or
    // require missing module facts, even for old aggregate-only records.
    for topology in [None, Some(generic_topology())] {
        for aggregate in [None, r.domains[0].executions[0].workspace.clone()] {
            let mut execution = r.domains[0].executions[0].clone();
            execution.execution_topology = topology.clone();
            execution.workspace = aggregate;
            execution.cache_update = CacheUpdateWorkspace::CopyState;
            let description =
                crate::workspace_resources::describe_text_workspace(&execution, &r, 17, 0, 4096)
                    .unwrap();
            let report = crate::resource_lifetimes::compose_resource_peaks(&description).unwrap();
            assert!(report.pools.is_empty());
            assert!(report.missing.is_empty());
            // Estimation still rejects an invalid calibration before attempting
            // to plan an invocation, as it did before retirement of the fallback.
            execution.workspace_overlap.upper_live_copies = Some(0);
            assert!(matches!(
                workspace(&execution, &r, 17, 0, 4096),
                Err(CapabilityError::InvalidConfiguration {
                    field: "workspace_overlap",
                    ..
                })
            ));
        }
    }
}

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
                state_layout,
                workspace: Some(WorkspaceGeometry {
                    hidden_size: 32,
                    intermediate_size: 64,
                    query_width: 32,
                    key_value_width: 8,
                    query_heads: 4,
                    vocabulary_size: 128,
                }),
                attention: AttentionWorkspace::Materialized,
                cache_update: CacheUpdateWorkspace::InPlace,
                logits: LogitsWorkspace::FinalPosition,
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
        logical_parameter_bytes: Observed::exact(100, "test"),
        current_host_resident_bytes: Observed::exact(100, "test"),
        current_device_resident_bytes: Observed::exact(100, "test"),
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

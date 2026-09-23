use super::*;
use eredu_core::{
    cache::LayerCachePolicy, AttentionPolicy, EstimationCompleteness, InputTokenCount,
    LayerSchedule, StateMemoryLayout,
};

fn request(domain: MemoryDomain) -> GenerationMemoryRequest {
    GenerationMemoryRequest {
        input: InputTokenCount::text(17),
        max_output_tokens: Some(32),
        forecast_output_tokens: 32,
        batch_size: 1,
        prefill_chunk_tokens: 3,
        scalar_bytes: NonZeroU8::new(4).unwrap(),
        domains: vec![DomainMemoryPlan {
            domain,
            resident_parameters: MemoryBytes::exact(4096),
            already_resident_bytes: 4096,
            retained_input: MemoryBytes::exact(68),
            staging: MemoryBytes::exact(0),
            backend_overhead: MemoryBytes::estimated(0, 65536, "shared allocator"),
            loading_peak: MemoryBytes::exact(0),
            budget: MemoryBudget {
                available_bytes: Some(1 << 30),
                ..Default::default()
            },
            executions: vec![ExecutionMemoryPlan {
                state_layout: StateMemoryLayout::new(
                    LayerSchedule::new(
                        1,
                        vec![LayerCachePolicy::key_only(AttentionPolicy::Full, 1, 8).unwrap()],
                    )
                    .unwrap(),
                    vec![0],
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
                attention: AttentionWorkspace::ScoreMatrixUpperBound,
                cache_update: CacheUpdateWorkspace::CopyState,
                logits: LogitsWorkspace::FinalPosition,
                workspace_overlap: WorkspaceOverlap::single_layer(),
            }],
        }],
    }
}

fn plan() -> SpeculativeMemoryPlan {
    SpeculativeMemoryPlan {
        draft: Some(request(MemoryDomain::Unified)),
        auxiliary_bytes_per_position: MemoryBytes::exact(0),
        sampling_bytes_per_vocabulary_entry: MemoryBytes::estimated(
            0,
            128,
            "fixture sampling scratch",
        ),
        max_draft_tokens: 4,
        scheduler: SpeculativeSchedulerOptions::default(),
        shared_allocator: true,
    }
}

fn upper(estimate: &GenerationMemoryEstimate) -> u64 {
    estimate.domains[0].generation_peak.upper_bytes.unwrap()
}

#[test]
fn bounded_phases_count_distinct_weights_once_and_share_allocator_overhead() {
    let request = request(MemoryDomain::Unified);
    let plan = plan();
    let estimate = estimate_speculative_memory(&request, &plan).unwrap();
    assert_eq!(estimate.fit, MemoryFit::LikelyFit);
    let domain = &estimate.domains[0];
    assert_eq!(domain.phases.len(), 5);
    for phase in domain
        .phases
        .iter()
        .filter(|p| p.phase != MemoryPhase::Loading)
    {
        assert_eq!(
            phase.parameters,
            MemoryBytes::exact(4096)
                .add(&MemoryBytes::exact(4096))
                .unwrap()
        );
        assert_eq!(phase.backend_overhead.upper_bytes, Some(65536));
        assert!(phase.staging.upper_bytes.unwrap() > 0);
    }
    assert_eq!(
        domain.phases[0].query_positions,
        request.input.model_positions
    );
    assert_eq!(domain.phases[2].phase, MemoryPhase::SpeculativeVerification);
    assert_eq!(domain.phases[2].query_positions, 5);
    assert_eq!(domain.phases[3].phase, MemoryPhase::SpeculativeCommit);
    assert_eq!(domain.phases[3].query_positions, 5);
    assert_eq!(
        domain.additional_generation_peak.upper_bytes,
        Some(upper(&estimate) - 8192)
    );
    assert_eq!(
        upper(&estimate),
        domain
            .phases
            .iter()
            .filter(|p| p.phase != MemoryPhase::Loading)
            .map(|p| p.total.upper_bytes.unwrap())
            .max()
            .unwrap()
    );
    // The caller cannot accidentally understate verification with ordinary final-row logits.
    let mut all_rows = request.clone();
    all_rows.domains[0].executions[0].logits = LogitsWorkspace::EveryPosition;
    assert_eq!(
        upper(&estimate),
        upper(&estimate_speculative_memory(&all_rows, &plan).unwrap())
    );
}

#[test]
fn configured_lookahead_and_proposal_width_raise_the_envelope() {
    let request = request(MemoryDomain::Unified);
    let mut plan = plan();
    let lookahead = upper(&estimate_speculative_memory(&request, &plan).unwrap());
    plan.scheduler = plan.scheduler.with_lookahead(false);
    let serial = upper(&estimate_speculative_memory(&request, &plan).unwrap());
    assert!(lookahead > serial);
    plan.max_draft_tokens = 1;
    let narrow = upper(&estimate_speculative_memory(&request, &plan).unwrap());
    assert!(serial > narrow);
    // These are global ceilings. One lane never submits two verifications or
    // creates two optimistic blocks merely because more lanes were permitted.
    plan.scheduler.max_in_flight_verifications = 10;
    plan.scheduler.max_optimistic_branches = 10;
    assert_eq!(
        narrow,
        upper(&estimate_speculative_memory(&request, &plan).unwrap())
    );
}

#[test]
fn missing_components_remain_unknown_without_erasing_known_residency() {
    let request = request(MemoryDomain::Unified);
    for missing in 0..4 {
        let mut plan = plan();
        match missing {
            0 => plan.auxiliary_bytes_per_position = MemoryBytes::unknown("prediction features"),
            1 => plan.sampling_bytes_per_vocabulary_entry = MemoryBytes::unknown("native sampler"),
            2 => plan.draft.as_mut().unwrap().domains[0].executions[0].workspace = None,
            _ => {
                plan.draft.as_mut().unwrap().domains[0].staging =
                    MemoryBytes::unknown("transfer overlap")
            }
        }
        let estimate = estimate_speculative_memory(&request, &plan).unwrap();
        assert_eq!(estimate.fit, MemoryFit::InsufficientInformation);
        assert!(estimate.domains[0].generation_peak.upper_bytes.is_none());
        assert!(estimate.domains[0].generation_peak.lower_bytes >= 8192);
    }
}

#[test]
fn embedded_ownership_does_not_charge_target_parameters_twice() {
    let request = request(MemoryDomain::Unified);
    let mut plan = plan();
    plan.draft = None;
    plan.auxiliary_bytes_per_position = MemoryBytes::exact(100);
    let estimate = estimate_speculative_memory(&request, &plan).unwrap();
    assert_eq!(estimate.domains[0].phases[0].parameters.lower_bytes, 4096);
    assert_eq!(
        estimate.domains[0].additional_generation_peak.upper_bytes,
        Some(upper(&estimate) - 4096)
    );
}

#[test]
fn shared_pool_uses_stricter_budgets_and_independent_devices_stay_separate() {
    let mut target = request(MemoryDomain::Unified);
    let mut plan = plan();
    plan.draft.as_mut().unwrap().domains[0]
        .budget
        .application_limit_bytes = Some(4096);
    assert_eq!(
        estimate_speculative_memory(&target, &plan).unwrap().fit,
        MemoryFit::LikelyShortfall
    );
    let draft = plan.draft.as_mut().unwrap();
    draft.domains[0].budget.application_limit_bytes = None;
    target.domains[0].domain = MemoryDomain::Device("target".into());
    draft.domains[0].domain = MemoryDomain::Device("draft".into());
    let mut host = request(MemoryDomain::Host).domains.remove(0);
    host.executions.clear();
    host.resident_parameters = MemoryBytes::exact(0);
    host.already_resident_bytes = 0;
    target.domains.push(host);
    let estimate = estimate_speculative_memory(&target, &plan).unwrap();
    assert_eq!(estimate.domains.len(), 3);
    assert_eq!(estimate.fit, MemoryFit::LikelyFit);
    assert_eq!(estimate.domains[0].phases[0].parameters.lower_bytes, 4096);
    assert_eq!(estimate.domains[2].phases[0].parameters.lower_bytes, 4096);
    assert!(estimate.domains[1].phases[0].staging.upper_bytes.unwrap() > 0);
    plan.draft.as_mut().unwrap().domains[0]
        .budget
        .available_bytes = Some(0);
    assert_ne!(
        estimate_speculative_memory(&target, &plan).unwrap().fit,
        MemoryFit::LikelyFit
    );
}

#[test]
fn overflow_invalid_intervals_and_incompatible_geometry_are_rejected() {
    let target = request(MemoryDomain::Unified);
    let mut plan = plan();
    plan.max_draft_tokens = u64::MAX;
    assert!(matches!(
        estimate_speculative_memory(&target, &plan),
        Err(CapabilityError::ArithmeticOverflow { .. })
    ));
    plan.max_draft_tokens = 4;
    plan.sampling_bytes_per_vocabulary_entry = MemoryBytes::estimated(100, 10, "invalid");
    assert!(estimate_speculative_memory(&target, &plan).is_err());
    plan.sampling_bytes_per_vocabulary_entry = MemoryBytes::exact(4);
    plan.draft.as_mut().unwrap().input.model_positions += 1;
    assert!(estimate_speculative_memory(&target, &plan).is_err());
    plan.draft = None;
    let mut target = target;
    target.batch_size = 2;
    assert!(estimate_speculative_memory(&target, &plan).is_err());
}

#[test]
fn plan_roundtrip_and_zero_output_keep_phase_accounting_coherent() {
    let mut target = request(MemoryDomain::Unified);
    let mut plan = plan();
    let decoded = serde_json::from_value(serde_json::to_value(&plan).unwrap()).unwrap();
    assert_eq!(plan, decoded);
    target.max_output_tokens = Some(0);
    target.forecast_output_tokens = 0;
    plan.draft = Some(target.clone());
    let estimate = estimate_speculative_memory(&target, &plan).unwrap();
    assert_eq!(estimate.domains[0].phases.len(), 2);
    assert_eq!(estimate.domains[0].phases[0].phase, MemoryPhase::Prefill);
    assert_eq!(estimate.domains[0].phases[1].phase, MemoryPhase::Loading);
}

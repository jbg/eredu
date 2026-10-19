use super::*;
use eredu_core::{
    cache::LayerCachePolicy, AttentionPolicy, EstimationCompleteness, InputTokenCount,
    LayerSchedule, StateMemoryLayout,
};

fn request(domain: MemoryDomain) -> GenerationMemoryRequest {
    GenerationMemoryRequest {
        parameter_conversion_retention: None,
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
                execution_topology: None,
                input_score_attention_mechanism: None,
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
                    gated_convolution: None,
                    input_score_attention: None,
                    mixed_precision_parameter_bytes: None,
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
        conditioned: None,
        shared_backbone: None,
        embedded: None,
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

fn conditioned() -> ConditionedAssistantMemoryPlan {
    ConditionedAssistantMemoryPlan {
        topology: ConditionedPredictionTopology {
            body: ConditionedPredictionBody {
                hidden: 4,
                captures: 2,
                layers: 2,
                query_heads: 2,
                head_width: 4,
                key_value_width: 2,
                intermediate: 8,
                context_window: 3,
                block_positions: 4,
                convolution: None,
            },
            scoring: ConditionedPredictionScoring::MappedVocabulary {
                vocabulary: 11,
                rank: 3,
            },
        },
        execution_domain: MemoryDomain::Unified,
        native_workspace: MemoryBytes::exact(0),
    }
}

#[test]
fn conditioned_context_compacts_but_capture_views_and_full_blocks_keep_their_backing() {
    let p = conditioned();
    let (state, workspace, features) = p.costs(10, 1, None).unwrap();
    // Three retained positions, each with a width-4 encoded row and two
    // width-2 key/value pairs. Capture suffix views can keep ten width-8 rows.
    assert_eq!(state.upper_bytes, Some(144));
    assert_eq!(features.upper_bytes, Some(320));
    assert_eq!(state.lower_bytes, 0);
    // Four complete width-11 raw score rows exist independently of the number
    // of candidates eventually sampled; they cannot disappear from the envelope.
    assert!(workspace.upper_bytes.unwrap() >= 4 * 11 * 4);
    let (long_state, _, long_features) = p.costs(100, 1, None).unwrap();
    assert_eq!(long_state, state);
    assert_eq!(long_features.upper_bytes, Some(3200));
    let mut convolution = p.clone();
    convolution.topology.body.convolution = Some(ConditionedBlockConvolution {
        taps: 3,
        group_size: 2,
    });
    let (conv_state, conv_work, _) = convolution.costs(10, 1, None).unwrap();
    assert_eq!(
        conv_state, state,
        "block-local convolution must not create persistent carry"
    );
    assert!(conv_work.upper_bytes.unwrap() > workspace.upper_bytes.unwrap());
    let mut malformed = p;
    malformed.topology.body.hidden = u64::MAX;
    assert!(malformed.costs(10, 1, None).is_err());
}

#[test]
fn conditioned_forecast_aggregates_one_parameter_owner_and_one_allocator_per_pool() {
    let target = request(MemoryDomain::Unified);
    let mut p = plan();
    p.max_draft_tokens = 2;
    p.conditioned = Some(conditioned());
    let draft = p.draft.as_mut().unwrap();
    draft.domains[0].executions.clear();
    draft.domains[0].resident_parameters = MemoryBytes::exact(3072);
    draft.domains[0].already_resident_bytes = 3072;
    draft.domains[0].backend_overhead = MemoryBytes::exact(32768);
    let estimate = estimate_speculative_memory(&target, &p).unwrap();
    assert!(estimate.domains[0]
        .phases
        .iter()
        .filter(|p| p.phase != MemoryPhase::Loading)
        .all(|p| p.parameters.lower_bytes == 7168
            && p.parameters.upper_bytes == Some(7168)
            && p.backend_overhead.upper_bytes == Some(65536)));
    let prefill = estimate.domains[0]
        .phases
        .iter()
        .find(|p| p.phase == MemoryPhase::Prefill)
        .unwrap();
    assert_eq!(
        prefill.retained_input.upper_bytes,
        Some(
            2 * (68 + std::mem::size_of::<eredu_core::GenerationSequence>() as u64 + 32 * 4)
                + 17 * 8 * 4
        ),
        "two declared host sequence/input allowances plus one shared capture backing envelope"
    );
    let mut short_target = target.clone();
    short_target.input = InputTokenCount::text(1);
    let mut short_plan = p.clone();
    short_plan.draft.as_mut().unwrap().input = short_target.input;
    assert_eq!(
        estimate_speculative_memory(&short_target, &short_plan)
            .unwrap()
            .domains[0]
            .state_growth_bytes_per_position,
        80,
        "32 target bytes plus the 48-byte conditioned FP32 context envelope before compaction"
    );
    p.conditioned.as_mut().unwrap().native_workspace =
        MemoryBytes::unknown("fixture native scratch unavailable");
    let unknown = estimate_speculative_memory(&target, &p).unwrap();
    assert_eq!(unknown.fit, MemoryFit::InsufficientInformation);
    assert!(unknown
        .uncertainties
        .iter()
        .any(|s| s == "fixture native scratch unavailable"));
    assert!(unknown.domains[0].generation_peak.upper_bytes.is_none());
    p.draft = None;
    assert!(estimate_speculative_memory(&target, &p).is_err());
}

#[test]
fn conditioned_split_pools_retain_target_capture_backing_before_transfer_and_rollback() {
    let source = MemoryDomain::Device("target".into());
    let destination = MemoryDomain::Device("assistant".into());
    let mut target = request(source.clone());
    target.max_output_tokens = Some(2);
    target.forecast_output_tokens = 2;
    let mut host = target.domains[0].clone();
    host.domain = MemoryDomain::Host;
    host.executions.clear();
    host.resident_parameters = MemoryBytes::exact(0);
    host.already_resident_bytes = 0;
    host.retained_input = MemoryBytes::exact(0);
    host.backend_overhead = MemoryBytes::exact(0);
    target.domains.push(host);
    let mut draft = target.clone();
    draft.domains.truncate(1);
    draft.domains[0].domain = destination.clone();
    draft.domains[0].executions.clear();
    draft.domains[0].retained_input = MemoryBytes::exact(0);
    let mut p = plan();
    p.max_draft_tokens = 2;
    p.draft = Some(draft);
    let mut assistant = conditioned();
    assistant.execution_domain = destination.clone();
    p.conditioned = Some(assistant);
    for lookahead in [false, true] {
        p.scheduler = p.scheduler.with_lookahead(lookahead);
        let estimate = estimate_speculative_memory(&target, &p).unwrap();
        let target_pool = estimate
            .domains
            .iter()
            .find(|d| d.domain == source)
            .unwrap();
        let prefill = target_pool
            .phases
            .iter()
            .find(|p| p.phase == MemoryPhase::Prefill)
            .unwrap();
        // Seventeen target rows, two ordered width-four captures, FP32 envelope.
        assert_eq!(
            prefill.retained_input.upper_bytes,
            Some(68 + 17 * 2 * 4 * 4)
        );
        assert!(prefill.staging.upper_bytes.unwrap() >= 2 * 17 * 2 * 4 * 4);
        let verification = target_pool
            .phases
            .iter()
            .find(|p| p.phase == MemoryPhase::SpeculativeVerification)
            .unwrap();
        // Two canonical outputs plus pending-token/proposal verification, with
        // another three positions when an optimistic block may remain live.
        let rows = if lookahead { 25 } else { 22 };
        assert_eq!(
            verification.retained_input.upper_bytes,
            Some(68 + rows * 2 * 4 * 4)
        );
        assert!(verification.staging.upper_bytes.unwrap() >= 2 * rows * 2 * 4 * 4);
        let draft_pool = estimate
            .domains
            .iter()
            .find(|d| d.domain == destination)
            .unwrap();
        let transferred = draft_pool
            .phases
            .iter()
            .find(|p| p.phase == MemoryPhase::Prefill)
            .unwrap();
        assert_eq!(transferred.retained_input.upper_bytes, Some(17 * 2 * 4 * 4));
        let host_pool = estimate
            .domains
            .iter()
            .find(|d| d.domain == MemoryDomain::Host)
            .unwrap();
        assert!(
            host_pool
                .phases
                .iter()
                .filter(|p| p.phase != MemoryPhase::Loading)
                .all(|p| p.retained_input.upper_bytes
                    == Some(std::mem::size_of::<eredu_core::GenerationSequence>() as u64 + 2 * 4)),
            "a metadata-only host pool owns the common sequence, without target publications"
        );
    }
}

#[test]
fn shortened_refinement_keeps_full_canvas_and_causal_prefix_separate() {
    let mut request = request(MemoryDomain::Unified);
    request.max_output_tokens = Some(1);
    let plan = RefinementMemoryPlan {
        canvas_positions: std::num::NonZeroU32::new(32).unwrap(),
        vocabulary_size: std::num::NonZeroU32::new(128).unwrap(),
        stability_history: 0,
        bytes_per_vocabulary_entry: MemoryBytes::estimated(0, 64, "numerical fixture"),
    };
    let estimate = estimate_refinement_memory(&request, &plan).unwrap();
    let domain = &estimate.domains[0];
    let refinement = domain
        .phases
        .iter()
        .find(|p| p.phase == MemoryPhase::Refinement)
        .unwrap();
    let commit = domain
        .phases
        .iter()
        .find(|p| p.phase == MemoryPhase::RefinementCommit)
        .unwrap();
    assert_eq!(refinement.query_positions, 32);
    // The selected cache rounds eighteen positions to its eight-position quantum.
    assert_eq!(refinement.persistent_state.lower_bytes, 24 * 8 * 4);
    assert_eq!(commit.query_positions, 1);
    assert_eq!(commit.positions, 18);
    assert!(refinement.workspace.upper_bytes.unwrap() >= 32 * 128 * 64 + 32 * 64 + 32 * 8 * 4);
    assert!(domain.phases.iter().all(|p| p.phase != MemoryPhase::Decode));
    assert!(domain
        .phases
        .iter()
        .filter(|p| p.phase != MemoryPhase::Loading)
        .all(|p| p.parameters.lower_bytes == 4096));
    request.max_output_tokens = Some(2);
    let longer = estimate_refinement_memory(&request, &plan).unwrap();
    assert_eq!(
        longer.domains[0]
            .phases
            .iter()
            .find(|p| p.phase == MemoryPhase::Refinement)
            .unwrap()
            .query_positions,
        32
    );
    assert!(upper(&longer) >= upper(&estimate));
    request.domains[0].loading_peak = MemoryBytes::exact(1 << 28);
    let source = SpeculativeMemoryPlan {
        conditioned: None,
        shared_backbone: None,
        embedded: None,
        draft: None,
        auxiliary_bytes_per_position: MemoryBytes::exact(0),
        sampling_bytes_per_vocabulary_entry: MemoryBytes::exact(64),
        max_draft_tokens: 0,
        scheduler: Default::default(),
        shared_allocator: true,
    };
    let combined = estimate_source_generation_memory(&request, &source, Some(&plan)).unwrap();
    assert!(combined.domains[0].overall_peak.lower_bytes >= 1 << 28);
    assert!(combined.domains[0].generation_peak.upper_bytes.unwrap() < 1 << 28);
    assert_eq!(
        combined.domains[0]
            .phases
            .iter()
            .find(|phase| phase.phase == MemoryPhase::Refinement)
            .unwrap()
            .parameters
            .lower_bytes,
        4096
    );
}

#[test]
fn queued_mode_forecasts_cover_all_handoffs_without_replaying_prefill_or_residency() {
    use eredu_core::decoding::{DecodingMode, DecodingState, SpeculationOptionsRequest};
    let refinement = RefinementMemoryPlan {
        canvas_positions: std::num::NonZeroU32::new(32).unwrap(),
        vocabulary_size: std::num::NonZeroU32::new(128).unwrap(),
        bytes_per_vocabulary_entry: MemoryBytes::estimated(0, 64, "numerical fixture"),
        stability_history: 0,
    };
    let mut speculative = plan();
    speculative.draft = None;
    speculative.shared_backbone = Some(SharedBackboneMemoryPlan {
        canvas_positions: 32,
        adapter: None,
    });
    let modes = [
        DecodingMode::Autoregressive,
        DecodingMode::Diffusion { policy: None },
        DecodingMode::SelfSpeculation(
            SpeculationOptionsRequest {
                max_draft_tokens: std::num::NonZeroUsize::new(4).unwrap(),
                ..Default::default()
            }
            .try_into()
            .unwrap(),
        ),
    ];
    let mut request = request(MemoryDomain::Unified);
    request.max_output_tokens = Some(2);
    request.forecast_output_tokens = 2;
    for active in modes {
        for pending in modes.into_iter().filter(|mode| *mode != active) {
            let mut decoding = DecodingContinuationMemoryPlan {
                state: DecodingState {
                    active,
                    pending: Some(pending),
                    acknowledgement: None,
                    terminal: false,
                },
                settlement_input_tokens: 0,
                refinement: vec![],
                speculative: vec![],
            };
            for mode in [active, pending] {
                match mode {
                    DecodingMode::Autoregressive => {}
                    DecodingMode::Diffusion { .. } => decoding.refinement.push(refinement.clone()),
                    DecodingMode::SelfSpeculation(_) => {
                        decoding.speculative.push(speculative.clone())
                    }
                }
            }
            let inputs = decoding.input_horizon(2, 2).unwrap();
            if !decoding.refinement.is_empty() {
                assert_eq!(inputs, 34);
            } else {
                assert!(inputs >= 7);
            }
            let native = ContinuationMemoryPlan {
                current_positions: 17,
                additional_input_tokens: inputs,
                query_tokens: 1,
                current_state: MemoryBytes::estimated(544, 1024, "installed capacity"),
                peak_state: MemoryBytes::estimated(544, 4096, "future capacity"),
            };
            let estimate =
                estimate_decoding_continuation_memory(&request, &native, &decoding).unwrap();
            let phases = &estimate.domains[0].phases;
            assert!(phases
                .iter()
                .all(|p| !matches!(p.phase, MemoryPhase::Loading | MemoryPhase::Prefill)));
            assert!(phases.iter().all(|p| p.parameters.lower_bytes == 4096));
            if !decoding.refinement.is_empty() {
                let phase = phases
                    .iter()
                    .find(|p| p.phase == MemoryPhase::Refinement)
                    .unwrap();
                assert_eq!(phase.query_positions, 32);
                assert_eq!(phase.persistent_state.upper_bytes, Some(4096));
                assert_eq!(
                    phases
                        .iter()
                        .find(|p| p.phase == MemoryPhase::RefinementCommit)
                        .unwrap()
                        .query_positions,
                    32
                );
            }
            if !decoding.speculative.is_empty() {
                let draft = phases
                    .iter()
                    .find(|p| p.phase == MemoryPhase::SpeculativeDraft)
                    .unwrap();
                assert_eq!(draft.query_positions, 32);
                assert!(draft.positions >= 17 + inputs + 32);
                assert!(phases
                    .iter()
                    .any(|p| p.phase == MemoryPhase::SpeculativeVerification));
            }
        }
    }
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
fn zero_output_keeps_prefill_and_loading_phases() {
    let mut target = request(MemoryDomain::Unified);
    let mut plan = plan();
    target.max_output_tokens = Some(0);
    target.forecast_output_tokens = 0;
    plan.draft = Some(target.clone());
    let estimate = estimate_speculative_memory(&target, &plan).unwrap();
    assert_eq!(estimate.domains[0].phases.len(), 2);
    assert_eq!(estimate.domains[0].phases[0].phase, MemoryPhase::Prefill);
    assert_eq!(estimate.domains[0].phases[1].phase, MemoryPhase::Loading);
}

fn continuation_fixture(
    tokens: u64,
    lookahead: bool,
) -> (
    GenerationMemoryRequest,
    SpeculativeMemoryPlan,
    SpeculativeContinuationMemoryPlan,
) {
    let mut target = request(MemoryDomain::Unified);
    let mut plan = plan();
    plan.scheduler = plan.scheduler.with_lookahead(lookahead);
    let additional =
        speculative_continuation_positions(tokens, plan.max_draft_tokens, plan.scheduler).unwrap();
    target.max_output_tokens = Some(additional);
    target.forecast_output_tokens = additional;
    target.domains[0].loading_peak = MemoryBytes::exact(u64::MAX / 4);
    let draft = plan.draft.as_mut().unwrap();
    draft.input = InputTokenCount::text(19);
    draft.max_output_tokens = Some(additional);
    draft.forecast_output_tokens = additional;
    draft.domains[0].loading_peak = MemoryBytes::unknown("completed draft loading");
    let state = |position| ContinuationMemoryPlan {
        current_positions: position,
        additional_input_tokens: additional,
        query_tokens: 1,
        current_state: MemoryBytes::estimated(0, 100_000, "retained capacity"),
        peak_state: MemoryBytes::estimated(0, 100_000 + additional * 100, "native capacity growth"),
    };
    (
        target,
        plan,
        SpeculativeContinuationMemoryPlan {
            additional_tokens: tokens,
            settlement_input_tokens: 0,
            target: state(17),
            draft: Some(state(19)),
            embedded: None,
            seed: MemoryBytes::estimated(0, 999, "seed"),
            host_retention: MemoryBytes::estimated(0, 888, "sampling and semantic state"),
            retained_snapshots: MemoryBytes::estimated(0, 777, "snapshots and branches"),
        },
    )
}

#[test]
fn conditioned_startup_geometry_cannot_substitute_for_installed_continuation_state() {
    let (target, mut plan, installed) = continuation_fixture(2, false);
    plan.conditioned = Some(conditioned());
    plan.draft.as_mut().unwrap().domains[0].executions.clear();
    assert!(matches!(
        estimate_speculative_continuation_memory(&target, &plan, &installed),
        Err(CapabilityError::InvalidConfiguration {
            field: "conditioned assistant continuation",
            ..
        })
    ));
}

#[test]
fn conditioned_continuation_keeps_installed_backing_and_unknown_future_capacity() {
    let (target, mut plan, mut installed) = continuation_fixture(2, false);
    plan.conditioned = Some(conditioned());
    plan.draft.as_mut().unwrap().domains[0].executions.clear();
    installed.draft.as_mut().unwrap().current_positions = 17;
    let estimate = estimate_speculative_continuation_memory(&target, &plan, &installed).unwrap();
    let phases = &estimate.domains[0].phases;
    assert_eq!(phases[0].phase, MemoryPhase::ContinuationStart);
    assert_eq!(phases[0].persistent_state.upper_bytes, Some(200_000));
    // Seven possible new positions: native target and context bounds each grow
    // by 700 bytes, plus 144 bytes for the three-position compact logical bank.
    assert_eq!(phases[1].persistent_state.upper_bytes, Some(201_544));
    assert!(phases.iter().all(|p| !matches!(p.phase, MemoryPhase::Loading | MemoryPhase::Prefill)));

    installed.draft.as_mut().unwrap().peak_state = MemoryBytes::unknown("future native backing capacity");
    let unknown = estimate_speculative_continuation_memory(&target, &plan, &installed).unwrap();
    assert_eq!(unknown.domains[0].phases[0].persistent_state.upper_bytes, Some(200_000));
    assert!(unknown.domains[0].generation_peak.upper_bytes.is_none());
}

#[test]
fn continuation_uses_both_installed_frontiers_and_capacity_without_completed_phases() {
    let (target, plan, continuation) = continuation_fixture(7, false);
    let result = estimate_speculative_continuation_memory(&target, &plan, &continuation).unwrap();
    assert_eq!(result.fit, MemoryFit::LikelyFit);
    assert_eq!(result.requested_positions, 24);
    let pool = &result.domains[0];
    assert_eq!(
        pool.phases.iter().map(|p| p.phase).collect::<Vec<_>>(),
        [
            MemoryPhase::ContinuationStart,
            MemoryPhase::SpeculativeDraft,
            MemoryPhase::SpeculativeVerification,
            MemoryPhase::SpeculativeCommit
        ]
    );
    let start = &pool.phases[0];
    assert_eq!(start.workspace.upper_bytes, Some(0));
    assert_eq!(start.persistent_state.upper_bytes, Some(200_000));
    assert!(start.persistent_state.lower_bytes > 0);
    assert_eq!(start.staging.upper_bytes, Some(999 + 777));
    assert_eq!(pool.phases[1].persistent_state.upper_bytes, Some(202_400));
    assert_eq!(
        pool.additional_generation_peak.upper_bytes,
        Some(pool.generation_peak.upper_bytes.unwrap() - 8192)
    );
    let restored: SpeculativeContinuationMemoryPlan =
        serde_json::from_str(&serde_json::to_string(&continuation).unwrap()).unwrap();
    assert_eq!(continuation, restored);
    assert_eq!(
        result,
        estimate_speculative_continuation_memory(&target, &plan, &restored).unwrap()
    );
}

#[test]
fn continuation_zero_horizon_retains_state_and_snapshots_but_no_transaction_workspace() {
    let (target, plan, mut continuation) = continuation_fixture(0, true);
    let before = estimate_speculative_continuation_memory(&target, &plan, &continuation).unwrap();
    assert_eq!(before.domains[0].phases.len(), 1);
    continuation.retained_snapshots = MemoryBytes::estimated(0, 1777, "extra child");
    let after = estimate_speculative_continuation_memory(&target, &plan, &continuation).unwrap();
    assert_eq!(upper(&after), upper(&before) + 1000);
    continuation.host_retention = MemoryBytes::unknown("unsupported custom sampler");
    let unknown = estimate_speculative_continuation_memory(&target, &plan, &continuation).unwrap();
    assert_eq!(unknown.fit, MemoryFit::InsufficientInformation);
    assert!(unknown.domains[0].generation_peak.lower_bytes > 8192);
}

#[test]
fn continuation_envelopes_grow_with_lookahead_and_reject_stale_horizons() {
    let (target, plan, continuation) = continuation_fixture(7, false);
    let base = estimate_speculative_continuation_memory(&target, &plan, &continuation).unwrap();
    let (look_target, look_plan, mut look) = continuation_fixture(7, true);
    assert!(
        upper(&estimate_speculative_continuation_memory(&look_target, &look_plan, &look).unwrap())
            > upper(&base)
    );
    look.additional_tokens += 1;
    assert!(estimate_speculative_continuation_memory(&look_target, &look_plan, &look).is_err());
    let mut embedded = plan.clone();
    embedded.draft = None;
    assert!(estimate_speculative_continuation_memory(&target, &embedded, &continuation).is_err());
    assert!(speculative_continuation_positions(u64::MAX, 4, plan.scheduler).is_err());
}

#[test]
fn continuation_unattributed_sampling_and_capture_storage_covers_discrete_pools() {
    let (mut target, mut plan, mut continuation) = continuation_fixture(0, false);
    let device = MemoryDomain::Device("gpu:0".into());
    target.domains[0].domain = device.clone();
    plan.draft.as_mut().unwrap().domains[0].domain = device;
    let mut host = target.domains[0].clone();
    host.domain = MemoryDomain::Host;
    host.executions.clear();
    host.resident_parameters = MemoryBytes::exact(0);
    host.already_resident_bytes = 0;
    target.domains.push(host);
    let known = estimate_speculative_continuation_memory(&target, &plan, &continuation).unwrap();
    for pool in &known.domains {
        assert!(pool.phases[0].retained_input.upper_bytes.unwrap() >= 888);
    }
    continuation.host_retention = MemoryBytes::unknown("unprojected native instrumentation");
    let unknown = estimate_speculative_continuation_memory(&target, &plan, &continuation).unwrap();
    for pool in &unknown.domains {
        assert!(pool.generation_peak.upper_bytes.is_none());
        assert_eq!(pool.fit, MemoryFit::InsufficientInformation);
    }
}

fn use_generic_topology(request: &mut GenerationMemoryRequest) {
    use crate::execution_topology::*;
    let projection = |name: &str, input, output| ProjectionTopology {
        arithmetic: eredu_nn::LinearArithmetic::Native,
        input,
        output,
        format: eredu_checkpoint::LinearFormat::Dense,
        bias: false,
        parameter: name.into(),
    };
    for domain in &mut request.domains {
        for execution in &mut domain.executions {
            execution.execution_topology = Some(TextExecutionTopology {
                iterative: None,
                hidden_size: 32,
                vocabulary_size: 128,
                selected_parameter_promotion_bytes: None,
                selected_parameter_promotion_payloads: Default::default(),
                projection_storage: Default::default(),
                projection_input_normalizations: Default::default(),
                f32_rms_normalization_gains: Default::default(),
                output_invocations: 1,
                output_softcap: false,
                output: projection("output", 32, 128),
                missing: vec![],
                layers: vec![TextLayerTopology {
                    input_projections: Vec::new(),
                    normalization_count: 2,
                    mixer: TokenMixerTopology::Attention {
                        query_heads: 4,
                        kv_heads: 1,
                        key_width: 8,
                        value_width: 8,
                        input_scores: false,
                        softcap: false,
                        sinks: false,
                        output_gate: false,
                        query_key_normalization: false,
                        rotary: true,
                        projections: vec![
                            projection("query", 32, 32),
                            projection("key", 32, 8),
                            projection("value", 32, 8),
                            projection("attention-output", 32, 32),
                        ],
                    },
                    feed_forward: FeedForwardTopology::Gated {
                        intermediate_size: 64,
                        projections: vec![
                            projection("gate", 32, 64),
                            projection("up", 32, 64),
                            projection("down", 64, 32),
                        ],
                    },
                }],
            });
            execution.workspace = None;
        }
    }
}

#[test]
fn speculative_startup_and_settled_continuation_use_generic_vocabulary_and_workspace() {
    let mut target = request(MemoryDomain::Unified);
    let mut plan = plan();
    use_generic_topology(&mut target);
    use_generic_topology(plan.draft.as_mut().unwrap());
    let startup = estimate_speculative_memory(&target, &plan).unwrap();
    assert_eq!(startup.fit, MemoryFit::LikelyFit);
    assert!(startup.domains[0].generation_peak.upper_bytes.is_some());
    let (mut target, mut plan, continuation) = continuation_fixture(7, false);
    use_generic_topology(&mut target);
    use_generic_topology(plan.draft.as_mut().unwrap());
    let before = (target.clone(), plan.clone(), continuation.clone());
    let settled = estimate_speculative_continuation_memory(&target, &plan, &continuation).unwrap();
    assert_eq!(settled.fit, MemoryFit::LikelyFit);
    assert!(settled.domains[0].generation_peak.upper_bytes.is_some());
    assert_eq!((target, plan, continuation), before);
}

fn embedded_fixture() -> (GenerationMemoryRequest, SpeculativeMemoryPlan) {
    use crate::prediction_resources::{PredictionExecutionMode, PredictionStateLayer};
    let mut target = request(MemoryDomain::Unified);
    use_generic_topology(&mut target);
    let execution = target.domains[0].executions[0].clone();
    let mut plan = plan();
    plan.draft = None;
    plan.embedded = Some(EmbeddedPredictionMemoryPlan {
        mode: PredictionExecutionMode::Sequential,
        state: vec![PredictionStateLayer {
            layer: 0,
            policy: LayerCachePolicy::key_only(AttentionPolicy::Full, 1, 8).unwrap(),
            processed_token_offset: -1,
        }],
        allocation_granularity: 8,
        execution,
        target_feature_bytes_per_position: MemoryBytes::exact(32 * 4),
        missing: vec![],
    });
    (target, plan)
}

#[test]
fn embedded_startup_is_bounded_with_one_parameter_owner_and_recomputable_horizons() {
    let (target, plan) = embedded_fixture();
    let before = (target.clone(), plan.clone());
    let estimate = estimate_speculative_memory(&target, &plan).unwrap();
    assert_eq!(estimate.fit, MemoryFit::LikelyFit);
    assert_eq!(estimate.domains[0].state_growth_bytes_per_position, 64);
    assert!(estimate.domains[0]
        .phases
        .iter()
        .filter(|p| p.phase != MemoryPhase::Loading)
        .all(|p| p.parameters == MemoryBytes::exact(4096)));
    assert_eq!(
        estimate.domains[0].additional_generation_peak.upper_bytes,
        Some(upper(&estimate) - 4096)
    );
    let mut serial = plan.clone();
    serial.scheduler = serial.scheduler.with_lookahead(false);
    assert!(upper(&estimate) > upper(&estimate_speculative_memory(&target, &serial).unwrap()));
    let mut longer = target.clone();
    longer.max_output_tokens = Some(128);
    assert!(upper(&estimate_speculative_memory(&longer, &plan).unwrap()) > upper(&estimate));
    let decoded: SpeculativeMemoryPlan =
        serde_json::from_str(&serde_json::to_string(&plan).unwrap()).unwrap();
    assert_eq!(
        estimate,
        estimate_speculative_memory(&target, &decoded).unwrap()
    );
    assert_eq!((target, plan), before);
}

#[test]
fn embedded_modes_use_ordinary_prefill_rows_and_zero_output_keeps_only_prefill() {
    use crate::prediction_resources::PredictionExecutionMode;
    let (mut target, mut plan) = embedded_fixture();
    target.input = InputTokenCount::text(1);
    target.max_output_tokens = Some(0);
    let embedded = plan.embedded.as_mut().unwrap();
    let (_, sequential, _) = embedded.costs(&target, 1, 1, true).unwrap();
    assert_eq!(sequential.upper_bytes, Some(0));
    embedded.mode = PredictionExecutionMode::Fused;
    let (_, fused, _) = embedded.costs(&target, 1, 1, true).unwrap();
    assert!(fused.upper_bytes.unwrap() > 0);
    let result = estimate_speculative_memory(&target, &plan).unwrap();
    assert_eq!(result.fit, MemoryFit::LikelyFit);
    assert_eq!(result.domains[0].phases.len(), 2);
    // Fused proposal row capacity is independent of the one declared state layer.
    target.max_output_tokens = Some(8);
    plan.max_draft_tokens = 1;
    let narrow = upper(&estimate_speculative_memory(&target, &plan).unwrap());
    plan.max_draft_tokens = 6;
    assert!(upper(&estimate_speculative_memory(&target, &plan).unwrap()) > narrow);
}

#[test]
fn embedded_missing_mechanisms_remain_named_and_known_residency_survives() {
    let (target, mut plan) = embedded_fixture();
    plan.embedded
        .as_mut()
        .unwrap()
        .missing
        .push("pooling attention native scratch".into());
    let result = estimate_speculative_memory(&target, &plan).unwrap();
    assert_eq!(result.fit, MemoryFit::InsufficientInformation);
    assert_eq!(result.domains[0].generation_peak.upper_bytes, None);
    assert!(result.domains[0].generation_peak.lower_bytes >= 4096);
    assert!(result
        .uncertainties
        .iter()
        .any(|r| r.contains("pooling attention native scratch")));
    plan.embedded.as_mut().unwrap().missing.clear();
    plan.embedded.as_mut().unwrap().execution.execution_topology = None;
    assert_eq!(
        estimate_speculative_memory(&target, &plan).unwrap().fit,
        MemoryFit::InsufficientInformation
    );
}

#[test]
fn embedded_discrete_pool_does_not_charge_prediction_storage_to_host() {
    let (mut target, plan) = embedded_fixture();
    target.domains[0].domain = MemoryDomain::Device("accelerator".into());
    let mut host = target.domains[0].clone();
    host.domain = MemoryDomain::Host;
    host.executions.clear();
    host.resident_parameters = MemoryBytes::exact(0);
    host.already_resident_bytes = 0;
    host.backend_overhead = MemoryBytes::exact(0);
    target.domains.push(host);
    let result = estimate_speculative_memory(&target, &plan).unwrap();
    assert_eq!(result.fit, MemoryFit::LikelyFit);
    assert!(
        result.domains[0].phases[0]
            .persistent_state
            .upper_bytes
            .unwrap()
            > 0
    );
    assert_eq!(
        result.domains[1].phases[0].persistent_state.upper_bytes,
        Some(0)
    );
    assert_eq!(result.domains[1].phases[0].workspace.upper_bytes, Some(0));
}

#[test]
fn embedded_invalid_ownership_continuation_and_overflow_are_rejected() {
    let (target, mut plan) = embedded_fixture();
    plan.draft = Some(target.clone());
    assert!(estimate_speculative_memory(&target, &plan).is_err());
    let (_, _, continuation) = continuation_fixture(1, false);
    assert!(estimate_speculative_continuation_memory(&target, &plan, &continuation).is_err());
    plan.draft = None;
    plan.embedded
        .as_mut()
        .unwrap()
        .target_feature_bytes_per_position = MemoryBytes::exact(u64::MAX);
    assert!(matches!(
        estimate_speculative_memory(&target, &plan),
        Err(CapabilityError::ArithmeticOverflow { .. })
    ));
}

#[test]
fn old_speculative_serialization_preserves_explicit_auxiliary_unknowns() {
    let mut plan = plan();
    plan.draft = None;
    plan.auxiliary_bytes_per_position = MemoryBytes::unknown("legacy unprojected predictor");
    let mut wire = serde_json::to_value(&plan).unwrap();
    wire.as_object_mut().unwrap().remove("embedded");
    let decoded: SpeculativeMemoryPlan = serde_json::from_value(wire).unwrap();
    assert_eq!(decoded, plan);
    assert_eq!(
        estimate_speculative_memory(&request(MemoryDomain::Unified), &decoded)
            .unwrap()
            .fit,
        MemoryFit::InsufficientInformation
    );
}

#[test]
fn embedded_conversion_credit_uses_authoritative_backing_bindings_without_new_residency() {
    use crate::{ResidentParameterConversion, ResidentParameterConversionBinding};
    use eredu_core::resources::*;
    let (mut target, mut plan) = embedded_fixture();
    let prediction = plan.embedded.as_mut().unwrap();
    for topology in [
        target.domains[0].executions[0]
            .execution_topology
            .as_mut()
            .unwrap(),
        prediction.execution.execution_topology.as_mut().unwrap(),
    ] {
        topology.selected_parameter_promotion_bytes = Some(300);
        topology.selected_parameter_promotion_payloads =
            [("shared".into(), 100), ("independent".into(), 200)].into();
    }
    let identity = |key: &str| ResourceIdentity {
        scope: "native-test".into(),
        key: key.into(),
    };
    let conversion = |key: &str, names: &[&str], bytes: u64| ResidentParameterConversion {
        allocation: ResourceAllocation {
            identity: identity(key),
            uses: vec![ResourceUse {
                owner: identity("owner"),
                role: ResourceRole::Parameters,
            }],
            placement: eredu_core::Observed::unavailable("no pool observation"),
            size: ResourceSize::Fixed {
                extent: ResourceExtent {
                    payload: ResourceByteBounds::exact(bytes),
                    capacity: ResourceByteBounds::unknown(bytes, "native capacity unknown"),
                },
            },
        },
        bindings: names
            .iter()
            .map(|name| ResidentParameterConversionBinding {
                retention_group: None,
                owner: identity("owner"),
                unit: eredu_core::residency::OffloadUnitId::new("unit").unwrap(),
                name: (*name).into(),
                logical_target: Some((*name).into()),
            })
            .collect(),
    };
    // One observed backing has two aliases. Only exact prepared destinations receive credit.
    let observed = vec![
        conversion("shared-allocation", &["shared", "alias"], 100),
        conversion("other-model", &["unrelated"], 200),
    ];
    let parameters = target.domains[0].resident_parameters.clone();
    let notes =
        apply_embedded_parameter_conversion_credit(&mut target, prediction, Some(&observed))
            .unwrap();
    assert!(notes[0].contains("300 bytes of distinct resident conversion backing"));
    assert_eq!(target.domains[0].resident_parameters, parameters);
    assert_eq!(
        target.domains[0].executions[0]
            .execution_topology
            .as_ref()
            .unwrap()
            .selected_parameter_promotion_bytes,
        Some(200)
    );
    assert_eq!(
        prediction
            .execution
            .execution_topology
            .as_ref()
            .unwrap()
            .selected_parameter_promotion_bytes,
        Some(200)
    );
    // Shared readout is absent from future work in both invocations; cached residency is unchanged.
    let before = (target.clone(), prediction.clone());
    apply_embedded_parameter_conversion_credit(&mut target, prediction, Some(&observed)).unwrap();
    assert_eq!((target.clone(), prediction.clone()), before);
    let duplicate = vec![observed[0].clone(), observed[0].clone()];
    assert!(
        apply_embedded_parameter_conversion_credit(&mut target, prediction, Some(&duplicate))
            .is_err()
    );
    assert_eq!((target.clone(), prediction.clone()), before);
    let independent = vec![
        conversion("first", &["independent"], 100),
        conversion("second", &["independent"], 100),
    ];
    apply_embedded_parameter_conversion_credit(&mut target, prediction, Some(&independent))
        .unwrap();
    assert_eq!((target.clone(), prediction.clone()), before);
    let complete = vec![conversion("complete", &["independent"], 200)];
    apply_embedded_parameter_conversion_credit(&mut target, prediction, Some(&complete)).unwrap();
    assert_eq!(
        prediction
            .execution
            .execution_topology
            .as_ref()
            .unwrap()
            .selected_parameter_promotion_bytes,
        Some(0)
    );
    assert_eq!(target.domains[0].resident_parameters, parameters);
}

fn embedded_continuation_fixture(
    tokens: u64,
    lookahead: bool,
) -> (
    GenerationMemoryRequest,
    SpeculativeMemoryPlan,
    SpeculativeContinuationMemoryPlan,
) {
    let (mut target, mut plan) = embedded_fixture();
    plan.scheduler = plan.scheduler.with_lookahead(lookahead);
    let extra =
        speculative_continuation_positions(tokens, plan.max_draft_tokens, plan.scheduler).unwrap();
    target.max_output_tokens = Some(extra);
    target.forecast_output_tokens = extra;
    target.domains[0].retained_input = MemoryBytes::exact(0);
    target.domains[0].loading_peak = MemoryBytes::unknown("completed loading");
    let continuation = SpeculativeContinuationMemoryPlan {
        settlement_input_tokens: 0,
        additional_tokens: tokens,
        target: ContinuationMemoryPlan {
            current_positions: 17,
            additional_input_tokens: extra,
            query_tokens: 1,
            current_state: MemoryBytes::estimated(0, 100_000, "installed target capacity"),
            peak_state: MemoryBytes::estimated(
                0,
                100_000 + extra * 100,
                "target native horizon capacity",
            ),
        },
        draft: None,
        embedded: Some(EmbeddedContinuationMemoryPlan {
            layer_positions: vec![13], // Actual owner wins over selected target-1 offset.
            additional_input_tokens: extra,
            current_state: MemoryBytes::estimated(0, 20_000, "installed prediction capacity"),
            peak_state: MemoryBytes::estimated(
                0,
                20_000 + extra * 40,
                "prediction native horizon capacity",
            ),
            retained_features: MemoryBytes::estimated(0, 128, "one-row view"),
        }),
        seed: MemoryBytes::estimated(0, 20_000, "current seed capacity"),
        host_retention: MemoryBytes::estimated(0, 1000, "semantic state"),
        retained_snapshots: MemoryBytes::exact(0),
    };
    (target, plan, continuation)
}

#[test]
fn embedded_continuation_uses_installed_layers_and_native_capacity_without_completed_phases() {
    let (target, plan, continuation) = embedded_continuation_fixture(7, false);
    let before = (target.clone(), plan.clone(), continuation.clone());
    let estimate = estimate_speculative_continuation_memory(&target, &plan, &continuation).unwrap();
    assert_eq!(estimate.fit, MemoryFit::LikelyFit);
    let start = &estimate.domains[0].phases[0];
    assert_eq!(start.phase, MemoryPhase::ContinuationStart);
    assert_eq!(start.persistent_state.lower_bytes, (17 + 13) * 8 * 4);
    assert_eq!(start.persistent_state.upper_bytes, Some(120_000));
    assert_eq!(start.workspace.upper_bytes, Some(0));
    assert!(estimate.domains[0]
        .phases
        .iter()
        .all(|p| p.phase != MemoryPhase::Loading && p.phase != MemoryPhase::Prefill));
    assert!(estimate.domains[0]
        .phases
        .iter()
        .all(|p| p.parameters == MemoryBytes::exact(4096)));
    assert_eq!(estimate.requested_positions, 24);
    let restored: SpeculativeContinuationMemoryPlan =
        serde_json::from_str(&serde_json::to_string(&continuation).unwrap()).unwrap();
    assert_eq!(
        estimate,
        estimate_speculative_continuation_memory(&target, &plan, &restored).unwrap()
    );
    assert_eq!((target, plan, continuation), before);
}

#[test]
fn embedded_continuation_zero_horizon_retains_prefix_backing_and_user_snapshots() {
    let (mut target, plan, mut continuation) = embedded_continuation_fixture(0, false);
    target.input = InputTokenCount::text(4096);
    continuation.target.current_positions = 4096;
    continuation.target.current_state.upper_bytes = Some(1_000_000);
    continuation.target.peak_state.upper_bytes = Some(1_000_000);
    let result = estimate_speculative_continuation_memory(&target, &plan, &continuation).unwrap();
    assert_eq!(result.fit, MemoryFit::LikelyFit);
    assert_eq!(result.domains[0].phases.len(), 1);
    // A retained last-row view may pin the complete prefill output. Its reported
    // logical copy bytes alone must never collapse that current upper bound.
    assert!(
        result.domains[0].phases[0]
            .retained_input
            .upper_bytes
            .unwrap()
            >= 4096 * 128
    );
    continuation.retained_snapshots = MemoryBytes::exact(12345);
    let reserved = estimate_speculative_continuation_memory(&target, &plan, &continuation).unwrap();
    assert_eq!(upper(&reserved), upper(&result) + 12345);
    assert_eq!(
        reserved.domains[0].additional_generation_peak.upper_bytes,
        Some(upper(&reserved) - 4096)
    );
}

#[test]
fn embedded_continuation_horizon_lookahead_and_unknowns_preserve_coverage() {
    let run = |tokens, lookahead| {
        let (target, plan, state) = embedded_continuation_fixture(tokens, lookahead);
        estimate_speculative_continuation_memory(&target, &plan, &state).unwrap()
    };
    assert!(upper(&run(30, false)) > upper(&run(1, false)));
    assert!(upper(&run(7, true)) > upper(&run(7, false)));
    let (target, mut plan, mut state) = embedded_continuation_fixture(7, false);
    state.embedded.as_mut().unwrap().peak_state =
        MemoryBytes::unknown("native prediction capacity unavailable");
    let estimate = estimate_speculative_continuation_memory(&target, &plan, &state).unwrap();
    assert_eq!(estimate.fit, MemoryFit::InsufficientInformation);
    assert!(estimate
        .uncertainties
        .iter()
        .any(|u| u.contains("native prediction capacity unavailable")));
    state.embedded.as_mut().unwrap().peak_state = MemoryBytes::estimated(0, 100_000, "capacity");
    plan.embedded
        .as_mut()
        .unwrap()
        .missing
        .push("custom prediction mechanism scratch".into());
    let estimate = estimate_speculative_continuation_memory(&target, &plan, &state).unwrap();
    assert_eq!(estimate.fit, MemoryFit::InsufficientInformation);
    assert!(estimate
        .uncertainties
        .iter()
        .any(|u| u.contains("custom prediction mechanism scratch")));
}

#[test]
fn embedded_continuation_rejects_mismatched_state_members_horizons_and_bounds() {
    let (target, plan, state) = embedded_continuation_fixture(7, false);
    for mutation in 0..5 {
        let mut state = state.clone();
        let embedded = state.embedded.as_mut().unwrap();
        match mutation {
            0 => embedded.layer_positions.push(0),
            1 => embedded.additional_input_tokens += 1,
            2 => embedded.layer_positions[0] = 18,
            3 => embedded.current_state.upper_bytes = Some(1),
            4 => {
                embedded.peak_state.upper_bytes =
                    Some(embedded.current_state.upper_bytes.unwrap() - 1)
            }
            _ => unreachable!(),
        }
        assert!(estimate_speculative_continuation_memory(&target, &plan, &state).is_err());
    }
}

#[test]
fn legacy_external_continuation_wire_remains_compatible_without_embedded_field() {
    let (target, plan, state) = continuation_fixture(7, false);
    let mut wire = serde_json::to_value(&state).unwrap();
    assert!(wire["draft"].is_object());
    wire.as_object_mut().unwrap().remove("embedded");
    let decoded = serde_json::from_value(wire).unwrap();
    assert_eq!(state, decoded);
    assert_eq!(
        estimate_speculative_continuation_memory(&target, &plan, &state).unwrap(),
        estimate_speculative_continuation_memory(&target, &plan, &decoded).unwrap()
    );
}

fn retention_fixture(group: &str, retained: u64, reserved: u64) -> ConversionRetentionMemoryPlan {
    use eredu_core::{residency::*, resources::*, Observed};
    let identity = |key: &str| ResourceIdentity {
        scope: "retention-fixture".into(),
        key: key.into(),
    };
    let group = ParameterConversionRetentionGroup(identity(group));
    let report = ParameterConversionRetentionReport {
        group: group.clone(),
        policy: Observed::exact(
            ParameterConversionRetentionPolicyReport::resolve(
                Some(ParameterConversionRetentionPolicy::Bounded { max_bytes: 1024 }),
                ParameterConversionRetentionEligibility::Eligible,
            ),
            "fixture",
        ),
        usage: Observed::exact(
            ParameterConversionRetentionUsage {
                retained_claims: u64::from(retained != 0),
                retained_payload_bytes: retained,
                reserved_payload_bytes: reserved,
                retained_backing_capacity_bytes: Observed::unavailable("padding unknown"),
            },
            "fixture",
        ),
    };
    let conversions = (retained != 0)
        .then(|| crate::ResidentParameterConversion {
            allocation: ResourceAllocation {
                identity: identity("shared-conversion"),
                uses: vec![ResourceUse {
                    owner: identity(&group.0.key),
                    role: ResourceRole::Parameters,
                }],
                placement: Observed::unavailable("single execution pool"),
                size: ResourceSize::Fixed {
                    extent: ResourceExtent {
                        payload: ResourceByteBounds::exact(retained),
                        capacity: ResourceByteBounds::unknown(retained, "padding unknown"),
                    },
                },
            },
            bindings: vec![crate::ResidentParameterConversionBinding {
                retention_group: Some(group.clone()),
                owner: identity(&group.0.key),
                unit: OffloadUnitId::new("unit").unwrap(),
                name: "shared".into(),
                logical_target: Some("shared".into()),
            }],
        })
        .into_iter()
        .collect();
    ConversionRetentionMemoryPlan::observe(&[report], Some(conversions), &[]).unwrap()
}

#[test]
fn shared_conversion_backing_is_deduplicated_but_independent_claims_survive() {
    let mut target = request(MemoryDomain::Unified);
    let mut p = plan();
    let historical = estimate_speculative_memory(&target, &p).unwrap();
    target.parameter_conversion_retention = Some(retention_fixture("target", 100, 0));
    p.draft.as_mut().unwrap().parameter_conversion_retention =
        Some(retention_fixture("draft", 100, 0));
    let composed = estimate_speculative_memory(&target, &p).unwrap();
    for phase in composed.domains[0]
        .phases
        .iter()
        .filter(|p| p.phase != MemoryPhase::Loading)
    {
        assert_eq!(phase.parameters.lower_bytes, 8192 - 100);
        assert_eq!(phase.parameters.upper_bytes, Some(8192 - 100));
    }
    assert_eq!(upper(&composed), upper(&historical) - 100);
    assert_eq!(
        composed.domains[0].additional_generation_peak,
        historical.domains[0].additional_generation_peak
    );
    assert_ne!(
        target
            .parameter_conversion_retention
            .as_ref()
            .unwrap()
            .groups[0]
            .report
            .group,
        p.draft
            .as_ref()
            .unwrap()
            .parameter_conversion_retention
            .as_ref()
            .unwrap()
            .groups[0]
            .report
            .group
    );
    // Trimming one independent claim removes only that model's residency. The
    // other participant continues to retain exactly one physical conversion.
    target.parameter_conversion_retention = Some(retention_fixture("target", 0, 0));
    target.domains[0].resident_parameters = MemoryBytes::exact(3996);
    target.domains[0].already_resident_bytes = 3996;
    let trimmed = estimate_speculative_memory(&target, &p).unwrap();
    assert_eq!(upper(&trimmed), upper(&composed));
    assert_eq!(
        trimmed.domains[0].additional_generation_peak,
        composed.domains[0].additional_generation_peak
    );
}

#[test]
fn finite_retention_does_not_clamp_temporary_workspace_or_cover_missing_mechanisms() {
    let mut request = request(MemoryDomain::Unified);
    request.domains[0].executions[0]
        .workspace
        .as_mut()
        .unwrap()
        .mixed_precision_parameter_bytes = Some(65536);
    let historical = estimate_generation_memory(&request).unwrap();
    request.parameter_conversion_retention = Some(retention_fixture("target", 0, 0));
    let bounded = estimate_generation_memory(&request).unwrap();
    assert_eq!(bounded.domains, historical.domains);
    assert!(bounded.domains[0].phases[0].workspace.upper_bytes.unwrap() > 1024);
    request.domains[0].executions[0].workspace_overlap = WorkspaceOverlap::unknown();
    assert!(estimate_generation_memory(&request).unwrap().domains[0]
        .generation_peak
        .upper_bytes
        .is_none());
    request.domains[0].executions[0].workspace = None;
    assert!(estimate_generation_memory(&request).unwrap().domains[0]
        .generation_peak
        .upper_bytes
        .is_none());
}

#[test]
fn reservation_publication_never_claims_unobserved_allocations_or_finite_overlap() {
    let mut request = request(MemoryDomain::Unified);
    request.parameter_conversion_retention = Some(retention_fixture("target", 0, 100));
    for (phase_name, query) in [
        (MemoryPhase::ContinuationStart, 0),
        (MemoryPhase::Decode, 1),
    ] {
        let phase = phase(&request.domains[0], &request, phase_name, 17, query).unwrap();
        assert_eq!(phase.parameters.lower_bytes, 4096);
        assert!(phase.parameters.upper_bytes.is_none());
        assert!(phase.total.upper_bytes.is_none());
    }
    request
        .parameter_conversion_retention
        .as_mut()
        .unwrap()
        .groups[0]
        .report
        .usage = eredu_core::Observed::unavailable("reservation usage was not observed");
    assert!(estimate_generation_memory(&request).unwrap().domains[0]
        .generation_peak
        .upper_bytes
        .is_none());
    request.parameter_conversion_retention = Some(retention_fixture("target", 100, 0));
    request.domains[0].resident_parameters = MemoryBytes::exact(4196);
    request.domains[0].already_resident_bytes = 4196;
    let settled = estimate_generation_memory(&request).unwrap();
    assert_eq!(settled.domains[0].phases[0].parameters.lower_bytes, 4196);
    assert!(settled.domains[0].generation_peak.upper_bytes.is_some());
}

#[test]
fn historical_requests_without_policy_recompute_their_original_accounting() {
    let original = request(MemoryDomain::Unified);
    let mut json = serde_json::to_value(&original).unwrap();
    json.as_object_mut()
        .unwrap()
        .remove("parameter_conversion_retention");
    let decoded: GenerationMemoryRequest = serde_json::from_value(json).unwrap();
    assert!(decoded.parameter_conversion_retention.is_none());
    assert_eq!(
        estimate_generation_memory(&decoded).unwrap(),
        estimate_generation_memory(&original).unwrap()
    );
    let mut current = original;
    current.parameter_conversion_retention = Some(retention_fixture("target", 100, 0));
    assert_eq!(
        serde_json::from_str::<GenerationMemoryRequest>(&serde_json::to_string(&current).unwrap())
            .unwrap(),
        current
    );
}

#[test]
fn shared_masked_forecast_keeps_the_full_canvas_and_one_parameter_owner() {
    let request = request(MemoryDomain::Unified);
    let mut plan = plan();
    plan.draft = None;
    plan.shared_backbone = Some(SharedBackboneMemoryPlan {
        canvas_positions: 32,
        adapter: None,
    });
    let full = estimate_speculative_memory(&request, &plan).unwrap();
    let proposal = full.domains[0]
        .phases
        .iter()
        .find(|phase| phase.phase == MemoryPhase::SpeculativeDraft)
        .unwrap();
    assert_eq!(proposal.query_positions, 32);
    assert_eq!(proposal.parameters.lower_bytes, 4096);
    let verification = full.domains[0]
        .phases
        .iter()
        .find(|phase| phase.phase == MemoryPhase::SpeculativeVerification)
        .unwrap();
    assert_eq!(verification.query_positions, 5);
    assert!(
        proposal.persistent_state.upper_bytes.unwrap()
            <= verification.persistent_state.upper_bytes.unwrap(),
        "masked block K/V does not enlarge canonical residency"
    );
    let workspace = proposal.workspace.upper_bytes.unwrap();
    plan.max_draft_tokens = 1;
    let shortened = estimate_speculative_memory(&request, &plan).unwrap();
    let proposal = shortened.domains[0]
        .phases
        .iter()
        .find(|phase| phase.phase == MemoryPhase::SpeculativeDraft)
        .unwrap();
    assert_eq!(proposal.query_positions, 32);
    assert_eq!(proposal.workspace.upper_bytes, Some(workspace));
    let nz = |n| std::num::NonZeroU32::new(n).unwrap();
    plan.shared_backbone.as_mut().unwrap().adapter = Some(
        crate::speculative::shared_backbone::SharedDraftAdapterGeometry {
            rank: nz(2),
            input: nz(32),
            output: nz(32),
            layers: nz(1),
        },
    );
    let adapted = estimate_speculative_memory(&request, &plan).unwrap();
    let adapted = adapted.domains[0]
        .phases
        .iter()
        .find(|phase| phase.phase == MemoryPhase::SpeculativeDraft)
        .unwrap();
    assert_eq!(adapted.parameters, proposal.parameters);
    let graphs = 1 + u64::from(
        plan.scheduler.lookahead_blocks != 0 && plan.scheduler.max_optimistic_branches != 0,
    );
    assert_eq!(
        adapted.workspace.upper_bytes.unwrap() - workspace,
        32 * (32 + 2 + 2 * 32) * 4 * graphs
    );
    assert!(
        adapted.staging.upper_bytes.unwrap() > 32 * 128 * 128,
        "full canvas distributions remain charged with shortened proposal budgets"
    );

    plan.max_draft_tokens = 0;
    let causal = estimate_speculative_memory(&request, &plan).unwrap();
    assert!(!causal.domains[0]
        .phases
        .iter()
        .any(|phase| phase.phase == MemoryPhase::SpeculativeDraft));
    let verification = causal.domains[0]
        .phases
        .iter()
        .find(|phase| phase.phase == MemoryPhase::SpeculativeVerification)
        .unwrap();
    assert_eq!(verification.query_positions, 1);
    assert_eq!(verification.parameters.lower_bytes, 4096);
    assert_eq!(
        speculative_continuation_positions(7, 0, plan.scheduler).unwrap(),
        8
    );
    let causal_peak = causal.domains[0].generation_peak.clone();
    plan.shared_backbone.as_mut().unwrap().adapter = None;
    assert_eq!(
        estimate_speculative_memory(&request, &plan)
            .unwrap()
            .domains[0]
            .generation_peak,
        causal_peak,
        "retained draft adapter has no causal invocation workspace"
    );
}

#[test]
fn queued_block_append_forecast_keeps_checkpoint_copies_with_zero_output() {
    use eredu_core::decoding::{DecodingMode, DecodingState};
    let mut request = request(MemoryDomain::Unified);
    request.max_output_tokens = Some(0);
    request.forecast_output_tokens = 0;
    let decoding = DecodingContinuationMemoryPlan {
        state: DecodingState {
            active: DecodingMode::Diffusion { policy: None },
            pending: Some(DecodingMode::Autoregressive),
            acknowledgement: None,
            terminal: false,
        },
        settlement_input_tokens: 32,
        speculative: vec![],
        refinement: vec![RefinementMemoryPlan {
            canvas_positions: std::num::NonZeroU32::new(32).unwrap(),
            vocabulary_size: std::num::NonZeroU32::new(128).unwrap(),
            stability_history: 0,
            bytes_per_vocabulary_entry: MemoryBytes::estimated(0, 64, "numerical fixture"),
        }],
    };
    assert_eq!(decoding.input_horizon(0, 0).unwrap(), 32);
    let native = ContinuationMemoryPlan {
        current_positions: 17,
        additional_input_tokens: 32,
        query_tokens: 1,
        current_state: MemoryBytes::estimated(544, 1024, "installed capacity"),
        peak_state: MemoryBytes::estimated(544, 4096, "future capacity"),
    };
    let estimate = estimate_decoding_continuation_memory(&request, &native, &decoding).unwrap();
    let phases = &estimate.domains[0].phases;
    let append = phases
        .iter()
        .find(|p| p.phase == MemoryPhase::SpeculativeVerification)
        .unwrap();
    assert_eq!(append.query_positions, 32);
    assert!(append.staging.upper_bytes.unwrap() >= 2 * 4096);
    assert!(phases
        .iter()
        .any(|p| p.phase == MemoryPhase::Refinement && p.query_positions == 32));
    assert!(phases.iter().all(|p| p.parameters.lower_bytes == 4096));
    assert!(phases.iter().all(|p| !matches!(
        p.phase,
        MemoryPhase::SpeculativeDraft | MemoryPhase::Loading | MemoryPhase::Prefill
    )));
}

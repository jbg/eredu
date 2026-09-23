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
        current_state: MemoryBytes::estimated(0, 100_000, "retained capacity"),
        peak_state: MemoryBytes::estimated(0, 100_000 + additional * 100, "native capacity growth"),
    };
    (
        target,
        plan,
        SpeculativeContinuationMemoryPlan {
            additional_tokens: tokens,
            target: state(17),
            draft: state(19),
            seed: MemoryBytes::estimated(0, 999, "seed"),
            host_retention: MemoryBytes::estimated(0, 888, "sampling and semantic state"),
            retained_snapshots: MemoryBytes::estimated(0, 777, "snapshots and branches"),
        },
    )
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
        input,
        output,
        format: eredu_checkpoint::LinearFormat::Dense,
        bias: false,
        parameter: name.into(),
    };
    for domain in &mut request.domains {
        for execution in &mut domain.executions {
            execution.execution_topology = Some(TextExecutionTopology {
                hidden_size: 32,
                vocabulary_size: 128,
                selected_parameter_promotion_bytes: None,
                selected_parameter_promotion_payloads: Default::default(),
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

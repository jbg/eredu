use super::*;
use eredu_core::{
    cache::LayerCachePolicy, capture::*, AttentionPolicy, EstimationCompleteness, InputTokenCount,
    LayerSchedule, StateMemoryLayout,
};

fn limits() -> CaptureLimits {
    CaptureLimits {
        per_step: CaptureUsage {
            captures: 1,
            retained_bytes: 100,
            host_bytes: 200,
            encoded_bytes: 300,
        },
        cumulative: CaptureUsage {
            captures: 10,
            retained_bytes: 1000,
            host_bytes: 2000,
            encoded_bytes: 3000,
        },
        physical_native_bytes: None,
        on_limit: CaptureLimitPolicy::Fail,
    }
}

fn trace() -> TraceLimits {
    TraceLimits {
        per_record_bytes: 500,
        total_bytes: 5000,
    }
}

fn request(domain: MemoryDomain) -> GenerationMemoryRequest {
    GenerationMemoryRequest {
        input: InputTokenCount::text(2),
        max_output_tokens: Some(2),
        forecast_output_tokens: 2,
        batch_size: 1,
        prefill_chunk_tokens: 2,
        scalar_bytes: NonZeroU8::new(4).unwrap(),
        domains: vec![DomainMemoryPlan {
            domain,
            resident_parameters: MemoryBytes::exact(100),
            already_resident_bytes: 100,
            retained_input: MemoryBytes::exact(8),
            staging: MemoryBytes::exact(0),
            backend_overhead: MemoryBytes::exact(0),
            loading_peak: MemoryBytes::exact(0),
            budget: MemoryBudget::default(),
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
                workspace: None,
                attention: AttentionWorkspace::Materialized,
                cache_update: CacheUpdateWorkspace::InPlace,
                logits: LogitsWorkspace::EveryPosition,
                workspace_overlap: WorkspaceOverlap::single_layer(),
            }],
        }],
    }
}

#[test]
fn native_lifetime_and_cumulative_host_retention_are_independent() {
    let (native, host) = bounds(&limits(), Some(400), trace(), true).unwrap();
    assert_eq!(native.upper_bytes, Some(100));
    assert_eq!(host.upper_bytes, Some(7400));
    let (native, same_host) = bounds(&limits(), Some(400), trace(), false).unwrap();
    assert_eq!(native.upper_bytes, Some(1000));
    assert_eq!(same_host, host);
    let mut smaller = limits();
    smaller.cumulative.retained_bytes = 50;
    assert_eq!(
        bounds(&smaller, Some(400), trace(), true)
            .unwrap()
            .0
            .upper_bytes,
        Some(50)
    );
    // A record can be captured before an independent trace budget rejects it.
    smaller.cumulative.encoded_bytes = 6000;
    assert_eq!(
        bounds(&smaller, Some(400), trace(), true)
            .unwrap()
            .1
            .upper_bytes,
        Some(8400)
    );
    smaller.on_limit = CaptureLimitPolicy::Skip;
    assert_eq!(
        bounds(&smaller, Some(400), trace(), true)
            .unwrap()
            .1
            .upper_bytes,
        Some(8400)
    );
}

#[test]
fn unified_separate_and_cpu_pools_preserve_existing_costs() {
    let (native, host) = bounds(&limits(), Some(400), trace(), true).unwrap();
    let mut unified = request(MemoryDomain::Unified);
    apply(&mut unified, &native, &host).unwrap();
    assert_eq!(unified.domains[0].retained_input.lower_bytes, 8);
    assert_eq!(unified.domains[0].retained_input.upper_bytes, Some(7508));
    assert_eq!(unified.domains[0].already_resident_bytes, 100);
    let mut cpu = request(MemoryDomain::Host);
    apply(&mut cpu, &native, &host).unwrap();
    assert_eq!(
        cpu.domains[0].retained_input,
        unified.domains[0].retained_input
    );
    let mut separate = request(MemoryDomain::Device("test".into()));
    let mut host_pool = request(MemoryDomain::Host).domains.remove(0);
    host_pool.executions.clear();
    separate.domains.push(host_pool);
    apply(&mut separate, &native, &host).unwrap();
    assert_eq!(separate.domains[0].retained_input.upper_bytes, Some(108));
    assert_eq!(separate.domains[1].retained_input.upper_bytes, Some(7408));
}

#[test]
fn missing_coverage_survives_and_arithmetic_never_wraps() {
    let (native, host) = bounds(&limits(), None, trace(), true).unwrap();
    assert!(host.upper_bytes.is_none());
    let mut unknown = request(MemoryDomain::Unified);
    unknown.domains[0].retained_input = MemoryBytes::unknown("unprojected media");
    let (_, known_host) = bounds(&limits(), Some(400), trace(), true).unwrap();
    apply(&mut unknown, &native, &known_host).unwrap();
    assert!(unknown.domains[0].retained_input.upper_bytes.is_none());
    assert!(unknown.domains[0]
        .retained_input
        .detail
        .contains("unprojected media"));
    let mut huge = limits();
    huge.cumulative.host_bytes = u64::MAX;
    assert!(matches!(
        bounds(&huge, Some(0), trace(), true),
        Err(CapabilityError::ArithmeticOverflow { .. })
    ));
    let mut overflow = request(MemoryDomain::Unified);
    overflow
        .domains
        .push(request(MemoryDomain::Unified).domains.remove(0));
    overflow.domains[1].retained_input = MemoryBytes::exact(u64::MAX);
    let before = overflow.clone();
    assert!(matches!(
        apply(&mut overflow, &native, &known_host),
        Err(CapabilityError::ArithmeticOverflow { .. })
    ));
    assert_eq!(overflow, before);
}

#[test]
fn admitted_intervention_payloads_are_included_in_host_storage() {
    use eredu_core::intervention::*;
    use eredu_core::{ObservationSupportStatus, SymbolicDimension, TensorAxis};
    let discovery = InterventionDiscovery {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        artifact_identity: "fixture".into(),
        session_identity: Some("backend-session".into()),
        points: vec![InterventionPoint {
            path: "block.output".into(),
            node_id: "block".into(),
            stage: InterventionStage::Activation,
            axes: vec![TensorAxis {
                name: "hidden".into(),
                dimension: SymbolicDimension::Known(4096),
            }],
            dtypes: vec![InterventionDtype::Float32],
            operations: vec![InterventionKind::Add],
            score_stages: vec![],
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            conditions: vec![],
            routed_units: None,
            routing: None,
        }],
    };
    let plan = InterventionPlan {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        operations: vec![InterventionOperation {
            id: "add".into(),
            target: "block.output".into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            action: InterventionAction::Add {
                tensor: InterventionTensor {
                    shape: vec![4096],
                    values: InterventionValues::Float32(vec![1.0; 4096]),
                },
            },
            evidence: InterventionEvidence::None,
        }],
    }
    .admit(
        &discovery,
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: 2,
            max_predictions: 2,
        },
        "session",
    )
    .unwrap();
    let storage = admitted_intervention_storage_bytes(&plan).unwrap();
    assert!(storage >= 4096 * 4);
    let (_, without) = bounds(&limits(), Some(0), trace(), true).unwrap();
    let (_, with) = bounds(&limits(), Some(storage), trace(), true).unwrap();
    assert!(with.upper_bytes.unwrap() >= without.upper_bytes.unwrap() + 4096 * 4);
}

fn projected_plan(request: &GenerationMemoryRequest, scoped: bool) -> CaptureMemoryPlan {
    use crate::capture::{CaptureUsageProjection, ScheduledCaptureUsage};
    let cost = CaptureUsage {
        captures: 1,
        retained_bytes: 20,
        host_bytes: 30,
        encoded_bytes: 40,
    };
    CaptureMemoryPlan {
        projection: Some(CaptureUsageProjection {
            first_prediction: 0,
            max_predictions: 10,
            per_step_metadata: CaptureUsage {
                host_bytes: 2,
                encoded_bytes: 3,
                ..Default::default()
            },
            selections: vec![ScheduledCaptureUsage {
                schedule: CaptureSchedule::default(),
                costs: [Some(cost); 2],
                required: false,
            }],
        }),
        first_prediction: 0,
        inherited_usage: CaptureUsage::default(),
        limits: limits(),
        trace: TraceLimits {
            per_record_bytes: 0,
            total_bytes: 0,
        },
        native_step_scoped: scoped,
        plan_storage: Some(10),
        baseline: request
            .domains
            .iter()
            .map(|d| (d.domain.clone(), d.retained_input.clone()))
            .collect(),
    }
}

#[test]
fn geometry_bounds_scale_with_horizon_not_quota_and_recompute_without_accumulating() {
    let mut request = request(MemoryDomain::Unified);
    let mut plan = projected_plan(&request, true);
    plan.apply(&mut request, 2).unwrap();
    assert_eq!(
        request.domains[0].retained_input.upper_bytes,
        Some(8 + 10 + 20 + 2 * (32 + 43))
    );
    let once = request.clone();
    plan.apply(&mut request, 2).unwrap();
    assert_eq!(request, once);
    plan.limits.cumulative.host_bytes *= 10;
    plan.limits.cumulative.retained_bytes *= 10;
    plan.apply(&mut request, 2).unwrap();
    assert_eq!(request, once);
    plan.apply(&mut request, 4).unwrap();
    assert_eq!(
        request.domains[0].retained_input.upper_bytes,
        Some(8 + 10 + 20 + 4 * 75)
    );
    plan.apply(&mut request, 0).unwrap();
    assert_eq!(request.domains[0].retained_input.upper_bytes, Some(18));
    plan.apply(&mut request, 11).unwrap();
    assert!(request.domains[0]
        .retained_input
        .detail
        .contains("admitted-limit fallback"));
    assert_eq!(
        request.domains[0].retained_input.upper_bytes,
        Some(18 + 100 + 20_000 + 3000)
    );
}

#[test]
fn continuation_keeps_charged_history_caps_remaining_costs_and_separates_pools() {
    let mut request = request(MemoryDomain::Device("test".into()));
    let mut host = request.domains[0].clone();
    host.domain = MemoryDomain::Host;
    host.executions.clear();
    request.domains.push(host);
    let mut plan = projected_plan(&request, true);
    plan.first_prediction = 3;
    plan.inherited_usage = CaptureUsage {
        captures: 3,
        retained_bytes: 990,
        host_bytes: 1990,
        encoded_bytes: 2990,
    };
    plan.apply(&mut request, 2).unwrap();
    assert_eq!(request.domains[0].retained_input.upper_bytes, Some(8 + 10)); // only 10 native bytes remain
    assert_eq!(
        request.domains[1].retained_input.upper_bytes,
        Some(8 + 10 + 2000 + 3000)
    );
    plan.apply(&mut request, 0).unwrap();
    assert_eq!(request.domains[0].retained_input.upper_bytes, Some(8));
    assert_eq!(
        request.domains[1].retained_input.upper_bytes,
        Some(8 + 10 + 1990 + 2990)
    );
    plan.native_step_scoped = false;
    plan.apply(&mut request, 2).unwrap();
    assert_eq!(request.domains[0].retained_input.upper_bytes, Some(1008));
    let before = request.clone();
    plan.first_prediction = u64::MAX;
    assert!(plan.apply(&mut request, 2).is_err());
    assert_eq!(request, before);
}

#[test]
fn unknown_transform_and_missing_projection_retain_the_quota_bound() {
    let mut request = request(MemoryDomain::Unified);
    let mut plan = projected_plan(&request, true);
    plan.projection.as_mut().unwrap().selections[0].costs[1] = None;
    plan.apply(&mut request, 2).unwrap();
    assert_eq!(
        request.domains[0].retained_input.upper_bytes,
        Some(18 + 100 + 2000 + 3000)
    );
    let bound = request.clone();
    plan.projection = None;
    plan.apply(&mut request, 2).unwrap();
    assert_eq!(request, bound);
}

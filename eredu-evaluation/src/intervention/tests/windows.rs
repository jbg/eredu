//! CaptureSession supplies controls; the existing Host and shared routing driver
//! remain the numerical oracle. No score equation is repeated in this fixture.
use super::*;
use eredu_core::*;
use eredu_runtime::{
    capture::{CaptureInvocationSelection, CaptureSession},
    RoutingDecision,
};
use std::sync::Arc;

struct Facts;
impl InterventionEstimator for Facts {
    fn original_route_usage(
        &self,
        _: &InterventionRoutingPolicy,
        _: u64,
    ) -> std::result::Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported(
            "no original-decision capture".into(),
        ))
    }
    fn validate_geometry(
        &self,
        _: &[u64],
        _: &ResolvedCaptureSlice,
    ) -> std::result::Result<(), CaptureError> {
        Ok(())
    }
    fn activation_usage(
        &self,
        _: &[u64],
        _: &ResolvedCaptureSlice,
        _: &InterventionAction,
    ) -> std::result::Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported("no dense transforms".into()))
    }
    fn capture_usage(
        &self,
        _: &[u64],
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> std::result::Result<CaptureUsage, CaptureError> {
        Err(CaptureError::Unsupported("no value evidence".into()))
    }
}
fn session(stage: RoutingScoreStage) -> CaptureSession {
    let bounds = CaptureInvocationBounds {
        batch: 1,
        max_sequence: 5,
        max_context: None,
        max_predictions: 1,
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        completeness: DescriptionCompleteness::Complete,
        points: vec![],
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: CaptureCapabilities::default(),
        points: vec![],
    };
    let usage = CaptureUsage {
        captures: 100,
        retained_bytes: 1 << 20,
        host_bytes: 1 << 20,
        encoded_bytes: 1 << 20,
    };
    let capture = CapturePlan {
        schema_version: 1,
        selections: vec![],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit_invocations(&catalog, &support, &support.capture, bounds)
    .unwrap();
    let point = InterventionPoint {
        path: "router".into(),
        node_id: "block".into(),
        stage: InterventionStage::RoutingBeforeDispatch,
        axes: vec![
            TensorAxis {
                name: "token".into(),
                dimension: SymbolicDimension::TokenRows,
            },
            TensorAxis {
                name: "selected_expert".into(),
                dimension: SymbolicDimension::Known(2),
            },
        ],
        dtypes: vec![],
        operations: vec![InterventionKind::BiasRoutingScores],
        score_stages: vec![
            RoutingScoreStage::RawLogits,
            RoutingScoreStage::TransformedScores,
            RoutingScoreStage::RankingScores,
        ],
        prefill: ObservationSupportStatus::Supported,
        decode: ObservationSupportStatus::Supported,
        conditions: vec![],
        routed_units: None,
        routing: Some(InterventionRoutingPolicy {
            expert_count: 4,
            top_k: 2,
            scoring: RoutingScoring::Softmax,
            normalize_selected: true,
            normalization_epsilon: 0.,
            coefficient_scale: 1.,
            groups: 1,
            selected_groups: 1,
            learned_coefficient_scale: false,
            shared_experts: 0,
        }),
    };
    let discovery = InterventionDiscovery {
        schema_version: 1,
        artifact_identity: "source".into(),
        session_identity: Some("loaded-session".into()),
        points: vec![point],
    };
    let plan = InterventionPlan {
        schema_version: 1,
        operations: vec![InterventionOperation {
            id: "bias".into(),
            target: "router".into(),
            schedule: CaptureSchedule::default(),
            slices: vec![CaptureSlice {
                axis: "token".into(),
                start: 1,
                end: 4,
                stride: 2,
            }],
            action: InterventionAction::BiasRoutingScores {
                stage,
                expert_ids: vec![0, 2],
                biases: vec![4., 0.25],
            },
            evidence: InterventionEvidence::None,
        }],
    }
    .admit_invocations(&discovery, bounds, "session")
    .unwrap();
    let mut session = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(capture));
    session.enable_interventions(plan, Arc::new(Facts)).unwrap();
    session
}
fn ordinary(host: &mut Host, input: &Value) -> GroupSelection<Value> {
    let raw = RoutingMechanism::project(host, input).unwrap();
    let scores = RoutingMechanism::transform(host, &raw).unwrap();
    let ranking = RoutingMechanism::ranking(host, &scores).unwrap();
    let ids = RoutingMechanism::select(host, &ranking).unwrap();
    RoutingMechanism::weights(host, &scores, ids).unwrap()
}
fn run(
    stage: RoutingScoreStage,
    windows: &[(u64, u64)],
    split: bool,
) -> (Vec<f32>, Vec<f32>, usize) {
    let mut host = Host::new();
    // Ranking-only corrections make the third bias boundary independently observable.
    host.correction = vec![0.01, 0.02, 0.03, 0.04];
    let mut session = session(stage);
    let (mut ids, mut weights) = (Vec::new(), Vec::new());
    for &(a, b) in windows {
        let shape = CaptureInvocationShape {
            batch: 1,
            sequence: b - a,
            context: None,
        };
        if split {
            session
                .begin_invocation_window(
                    CapturePhase::Prefill,
                    0,
                    shape,
                    CaptureInvocationSelection::default(),
                    CaptureInvocationWindow {
                        logical_sequence: 5,
                        start: a,
                    },
                )
                .unwrap();
        } else {
            session
                .begin_invocation(
                    CapturePhase::Prefill,
                    0,
                    shape,
                    CaptureInvocationSelection::default(),
                )
                .unwrap();
        }
        let input = Value::f32(
            &[b - a, 4],
            (a..b)
                .flat_map(|r| [0.1 + r as f32 * 0.07, 1.2, 2.3 - r as f32 * 0.05, 3.4])
                .collect(),
        );
        let control = session.routing_control("router", b - a).unwrap();
        let selected = if let Some(control) = control {
            assert_eq!(control.row_stride, 2);
            let actual = execute_routing_intervention(&mut host, &input, &control).unwrap();
            session
                .routing_applied(
                    &mut host,
                    "router",
                    actual.original.as_ref().map(|v| RoutingDecision {
                        ids: v.group_indices(),
                        coefficients: v.coefficients(),
                    }),
                    RoutingDecision {
                        ids: actual.effective.group_indices(),
                        coefficients: actual.effective.coefficients(),
                    },
                )
                .unwrap();
            actual.effective
        } else {
            assert!(
                a == 0 || a == 4,
                "only genuine non-overlap windows use ordinary selection"
            );
            assert_eq!(
                session.routing_unmodified_interest("router"),
                eredu_runtime::RoutingUnmodifiedInterest::Metadata
            );
            let actual = ordinary(&mut host, &input);
            session
                .routing_unmodified(
                    &mut host,
                    "router",
                    RoutingDecision {
                        ids: actual.group_indices(),
                        coefficients: actual.coefficients(),
                    },
                )
                .unwrap();
            actual
        };
        ids.extend_from_slice(&selected.group_indices().values);
        weights.extend_from_slice(&selected.coefficients().values);
        session.finish_interventions().unwrap();
        let step = session.take_shared_step().unwrap().as_step().clone();
        assert_eq!(
            step.interventions[0].outcome,
            if split && (a == 0 || a == 4) {
                InterventionOutcome::Unmatched
            } else {
                InterventionOutcome::Applied
            }
        );
    }
    assert_eq!(host.projections, windows.len());
    (ids, weights, host.projections)
}
#[test]
fn capture_projected_bias_rows_match_shared_numerical_worker_at_all_score_stages() {
    for stage in [
        RoutingScoreStage::RawLogits,
        RoutingScoreStage::TransformedScores,
        RoutingScoreStage::RankingScores,
    ] {
        let full = run(stage, &[(0, 5)], false);
        let split = run(stage, &[(0, 1), (1, 3), (3, 4), (4, 5)], true);
        assert_eq!(full.0, split.0);
        assert_eq!(full.1, split.1);
        assert_eq!((full.2, split.2), (1, 4));
        for row in [0, 2, 4] {
            assert_eq!(&split.0[row * 2..row * 2 + 2], &[3., 2.]);
        }
        for row in [1, 3] {
            assert_eq!(
                split.0[row * 2],
                0.,
                "bias must affect the selected global row"
            );
        }
        assert!(split.1.iter().all(|v| v.is_finite() && *v > 0.));
    }
}

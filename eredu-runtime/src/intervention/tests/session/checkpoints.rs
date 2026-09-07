use super::*;
use crate::capture::{CaptureForkRequest, InterventionForkRequest};

fn captures(plan: &AdmittedCapturePlan) -> CaptureDiscovery {
    CaptureDiscovery {
        artifact_identity: "source".into(),
        catalog: ObservationCatalog {
            schema_version: 1,
            completeness: DescriptionCompleteness::Complete,
            points: plan.points().to_vec(),
        },
        support: ObservationSupportReport {
            schema_version: 1,
            points: plan
                .points()
                .iter()
                .map(|p| ObservationSupport {
                    path: p.path.clone(),
                    prefill: ObservationSupportStatus::Supported,
                    decode: ObservationSupportStatus::Supported,
                    floating_to_f32: true,
                })
                .collect(),
            capture: CaptureCapabilities {
                transformations: vec![CaptureTransformKind::Preview],
                ..Default::default()
            },
        },
    }
}

fn advance(session: &mut CaptureSession, prediction: u64, ordinary: bool) -> CapturedStep {
    let input = Value {
        shape: vec![if prediction == 0 { 2 } else { 1 }, 2],
        data: vec![1.; if prediction == 0 { 4 } else { 2 }],
    };
    session
        .begin_step(
            if prediction == 0 {
                CapturePhase::Prefill
            } else {
                CapturePhase::Decode
            },
            prediction,
        )
        .unwrap();
    let mut backend = Backend::default();
    if ordinary {
        session
            .observe(&mut backend, "block.output", &input)
            .unwrap();
    }
    session
        .intervene(&mut backend, "block.output", &input)
        .unwrap();
    session.finish_interventions().unwrap();
    session.take_step().unwrap()
}

#[test]
fn combined_and_intervention_only_checkpoints_rebind_children_and_retain_outcomes() {
    for ordinary in [false, true] {
        let (capture, intervention) = plans(
            vec![operation(
                "zero",
                InterventionAction::Zero {
                    dtype: InterventionDtype::Float32,
                },
            )],
            ordinary,
        );
        let source = captures(&capture);
        let targets = discovery(&intervention);
        let mut parent = CaptureSession::new(capture.clone());
        parent
            .enable_interventions(intervention.clone(), Arc::new(Facts::new(0)))
            .unwrap();
        advance(&mut parent, 0, ordinary);
        let saved = parent.checkpoint(&source).unwrap();
        let mut child = saved
            .fork(
                CaptureForkRequest {
                    discovery: &source,
                    max_predictions: 3,
                    limits: capture.plan().limits.clone(),
                    intervention: Some(InterventionForkRequest {
                        discovery: &targets,
                        session_id: "child",
                        replacement: None,
                        estimator: Arc::new(Facts::new(0)),
                    }),
                },
                estimate,
            )
            .unwrap();
        let baseline = advance(&mut parent, 1, ordinary);
        let result = advance(&mut child, 1, ordinary);
        assert_eq!(result.prediction_index, baseline.prediction_index);
        assert_eq!(
            result.interventions[0].outcome,
            InterventionOutcome::Applied
        );
        assert_eq!(
            result.interventions[0].evidence,
            baseline.interventions[0].evidence
        );
        assert_ne!(
            result.interventions[0].plan_id,
            baseline.interventions[0].plan_id
        );
        assert_eq!(
            saved.intervention_plan().unwrap().identity(),
            intervention.identity()
        );
        let spent = parent.cumulative_usage();
        parent.restore(&saved).unwrap();
        assert!(parent.take_step().is_none());
        assert_eq!(parent.cumulative_usage(), spent);
        assert_eq!(
            advance(&mut parent, 1, ordinary).interventions,
            baseline.interventions
        );
    }
}

#[test]
fn fork_rejects_copied_identity_absent_revalidation_and_unavailable_native_estimates() {
    let (capture, plan) = routed(true);
    let source = captures(&capture);
    let targets = discovery(&plan);
    let mut parent = CaptureSession::new(capture.clone());
    parent
        .enable_interventions(plan.clone(), Arc::new(Facts::new(1)))
        .unwrap();
    let saved = parent.checkpoint(&source).unwrap();
    assert!(saved
        .fork(
            CaptureForkRequest {
                discovery: &source,
                max_predictions: 3,
                limits: capture.plan().limits.clone(),
                intervention: None,
            },
            estimate
        )
        .is_err());
    let fork = |session_id, estimator: Facts| {
        saved.fork(
            CaptureForkRequest {
                discovery: &source,
                max_predictions: 3,
                limits: capture.plan().limits.clone(),
                intervention: Some(InterventionForkRequest {
                    discovery: &targets,
                    session_id,
                    replacement: None,
                    estimator: Arc::new(estimator),
                }),
            },
            estimate,
        )
    };
    assert!(fork(plan.session_id(), Facts::new(1)).is_err());
    let mut unavailable = Facts::new(1);
    unavailable.unavailable = true;
    assert!(fork("child", unavailable).is_err());
    assert!(fork("child", Facts::new(1)).is_ok());
    parent.begin_step(CapturePhase::Prefill, 0).unwrap();
    parent.routing_control("router", 2).unwrap().unwrap();
    assert!(parent.finish_interventions().is_err());
    parent.take_step().unwrap();
    assert!(parent.checkpoint(&source).is_err());
    assert!(parent.restore(&saved).is_err());
}

#[test]
fn prospective_removal_keeps_inherited_accounting_and_original_provenance() {
    let (capture, plan) = plans(
        vec![operation(
            "zero",
            InterventionAction::Zero {
                dtype: InterventionDtype::Float32,
            },
        )],
        false,
    );
    let source = captures(&capture);
    let targets = discovery(&plan);
    let mut parent = CaptureSession::new(capture.clone());
    parent
        .enable_interventions(plan.clone(), Arc::new(Facts::new(0)))
        .unwrap();
    advance(&mut parent, 0, false);
    let saved = parent.checkpoint(&source).unwrap();
    let mut child = saved
        .fork(
            CaptureForkRequest {
                discovery: &source,
                max_predictions: 3,
                limits: capture.plan().limits.clone(),
                intervention: Some(InterventionForkRequest {
                    discovery: &targets,
                    session_id: "child",
                    replacement: Some(InterventionPlan::none()),
                    estimator: Arc::new(Facts::new(0)),
                }),
            },
            estimate,
        )
        .unwrap();
    assert_eq!(child.cumulative_usage(), saved.inherited_usage());
    child.begin_step(CapturePhase::Decode, 1).unwrap();
    let record = child.take_step().unwrap();
    assert!(record.interventions.is_empty());
    assert_eq!(record.cumulative_usage, saved.inherited_usage());
    assert_eq!(
        saved.intervention_plan().unwrap().identity(),
        plan.identity()
    );
    assert_eq!(
        advance(&mut parent, 1, false).interventions[0].outcome,
        InterventionOutcome::Applied
    );
}

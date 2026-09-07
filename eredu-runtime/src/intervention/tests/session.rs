use super::*;
use std::sync::{Arc, Mutex};

mod checkpoints;

struct Facts {
    max_index: u64,
    bytes_per_row: u64,
    unavailable: bool,
    calls: Mutex<Vec<u64>>,
}
impl Facts {
    fn new(bytes: u64) -> Self {
        Self {
            max_index: u64::MAX,
            bytes_per_row: bytes,
            unavailable: false,
            calls: Mutex::new(vec![]),
        }
    }
}
impl InterventionEstimator for Facts {
    fn validate_geometry(
        &self,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
    ) -> Result<(), CaptureError> {
        if elements(source)? > self.max_index || slice.strides.iter().any(|v| *v > self.max_index) {
            return Err(CaptureError::Unsupported("backend index limit".into()));
        }
        Ok(())
    }
    fn capture_usage(
        &self,
        source: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        estimate(source, selection, slice)
    }
    fn original_route_usage(
        &self,
        _: &InterventionRoutingPolicy,
        rows: u64,
    ) -> Result<CaptureUsage, CaptureError> {
        self.calls.lock().unwrap().push(rows);
        if self.unavailable {
            return Err(CaptureError::Unsupported(
                "original decision estimate unavailable".into(),
            ));
        }
        Ok(CaptureUsage {
            retained_bytes: mul(rows, self.bytes_per_row)?,
            host_bytes: mul(rows, 8)?,
            ..Default::default()
        })
    }
}
fn discovery(plan: &AdmittedInterventionPlan) -> InterventionDiscovery {
    InterventionDiscovery {
        schema_version: 1,
        artifact_identity: plan.artifact_identity().into(),
        session_identity: Some("backend-session".into()),
        points: plan.points().to_vec(),
    }
}
fn routed(evidence: bool) -> (AdmittedCapturePlan, AdmittedInterventionPlan) {
    let (capture, _) = plans(vec![], false);
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
        operations: vec![InterventionKind::ExcludeExperts],
        score_stages: vec![],
        prefill: ObservationSupportStatus::Supported,
        decode: ObservationSupportStatus::Supported,
        conditions: vec![],
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
            shared_experts: 1,
        }),
    };
    let catalog = InterventionDiscovery {
        schema_version: 1,
        artifact_identity: "source".into(),
        session_identity: Some("backend-session".into()),
        points: vec![point],
    };
    let plan = InterventionPlan {
        schema_version: 1,
        operations: vec![InterventionOperation {
            id: "routes".into(),
            target: "router".into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            action: InterventionAction::ExcludeExperts {
                expert_ids: vec![3],
            },
            evidence: if evidence {
                InterventionEvidence::Preview { max_elements: 2 }
            } else {
                InterventionEvidence::None
            },
        }],
    }
    .admit(&catalog, capture.request(), "session")
    .unwrap();
    (capture, plan)
}
fn limited(capture: &AdmittedCapturePlan, retained: u64) -> AdmittedCapturePlan {
    let mut plan = capture.plan().clone();
    plan.limits.per_step.retained_bytes = retained;
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
    plan.admit(&catalog, &support, &support.capture, capture.request())
        .unwrap()
}

#[test]
fn shared_admission_uses_backend_limits_and_loaded_session_identity() {
    let (capture, plan) = plans(
        vec![operation(
            "zero",
            InterventionAction::Zero {
                dtype: InterventionDtype::Float32,
            },
        )],
        false,
    );
    let mut facts = Facts::new(0);
    facts.max_index = 3;
    assert!(validate_session(&capture, &plan, &discovery(&plan), &facts).is_err());
    facts.max_index = 4;
    validate_session(&capture, &plan, &discovery(&plan), &facts).unwrap();
    let mut other = discovery(&plan);
    other.session_identity = Some("another-loaded-session".into());
    assert!(validate_session(&capture, &plan, &other, &facts).is_err());
    let mut slot = None;
    install_session(
        &mut slot,
        capture.clone(),
        Some((plan.clone(), Arc::new(facts))),
    )
    .unwrap();
    assert!(install_session(&mut slot, capture, None).is_err());
    slot.as_mut()
        .unwrap()
        .begin_step(CapturePhase::Prefill, 0)
        .unwrap();
    assert!(slot
        .as_mut()
        .unwrap()
        .enable_interventions(plan, Arc::new(Facts::new(0)))
        .is_err());
}

#[test]
fn original_estimates_are_used_at_admission_and_reserved_before_routing() {
    let (capture, plan) = routed(true);
    for bytes in [7, 7000] {
        let facts = Arc::new(Facts::new(bytes));
        validate_session(&capture, &plan, &discovery(&plan), facts.as_ref()).unwrap();
        assert_eq!(*facts.calls.lock().unwrap(), [2, 1]);
        let mut session = CaptureSession::new(capture.clone());
        session
            .enable_interventions(plan.clone(), facts.clone())
            .unwrap();
        let mut backend = Backend::default();
        for (phase, prediction, rows) in
            [(CapturePhase::Prefill, 0, 2), (CapturePhase::Decode, 1, 1)]
        {
            session.begin_step(phase, prediction).unwrap();
            let control = session.routing_control("router", rows).unwrap().unwrap();
            assert!(control.capture_original);
            let ids = Value {
                shape: vec![rows, 2],
                data: vec![1.; rows as usize * 2],
            };
            let weights = Value {
                shape: vec![rows, 2],
                data: vec![0.5; rows as usize * 2],
            };
            session
                .routing_applied(
                    &mut backend,
                    "router",
                    Some(crate::RoutingDecision {
                        ids: &ids,
                        coefficients: &weights,
                    }),
                    crate::RoutingDecision {
                        ids: &ids,
                        coefficients: &weights,
                    },
                )
                .unwrap();
            session.finish_interventions().unwrap();
            let step = session.take_step().unwrap();
            assert_eq!(step.interventions[0].charged.retained_bytes, rows * bytes);
            assert_eq!(
                step.step_usage.retained_bytes,
                rows * bytes + 4 * 128,
                "original work is distinct from four evidence transforms"
            );
            assert_eq!(step.step_usage.captures, 4);
            assert_eq!(step.interventions[0].outcome, InterventionOutcome::Applied);
        }
        assert_eq!(*facts.calls.lock().unwrap(), [2, 1, 2, 1]);
    }
}

#[test]
fn unavailable_overflowing_and_over_budget_estimates_fail_before_extra_work() {
    let (capture, plan) = routed(true);
    let mut unavailable = Facts::new(1);
    unavailable.unavailable = true;
    for facts in [unavailable, Facts::new(u64::MAX), Facts::new(2_000_000)] {
        let facts = Arc::new(facts);
        assert!(preflight(&capture, &plan, facts.as_ref()).is_err());
        // Even a direct caller bypassing cold preflight receives a hard failure
        // before a selector can obtain an executable routing control.
        let mut session = CaptureSession::new(capture.clone());
        session.enable_interventions(plan.clone(), facts).unwrap();
        session.begin_step(CapturePhase::Prefill, 0).unwrap();
        assert!(session.routing_control("router", 2).is_err());
        assert!(session.finish_interventions().is_err());
        assert!(matches!(
            session.take_step().unwrap().interventions[0].outcome,
            InterventionOutcome::Failed { .. }
        ));
    }
    let expensive = Facts::new(1000);
    assert!(preflight(&limited(&capture, 1000), &plan, &expensive).is_err());
    preflight(&limited(&capture, 3000), &plan, &expensive).unwrap();
}

#[test]
fn no_original_evidence_never_requests_an_estimate_or_original_decision() {
    let (capture, plan) = routed(false);
    let mut unavailable = Facts::new(0);
    unavailable.unavailable = true;
    let facts = Arc::new(unavailable);
    preflight(&capture, &plan, facts.as_ref()).unwrap();
    let mut session = CaptureSession::new(capture);
    session.enable_interventions(plan, facts.clone()).unwrap();
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    assert!(
        !session
            .routing_control("router", 2)
            .unwrap()
            .unwrap()
            .capture_original
    );
    let values = Value {
        shape: vec![2, 2],
        data: vec![1.; 4],
    };
    session
        .routing_applied(
            &mut Backend::default(),
            "router",
            None,
            crate::RoutingDecision {
                ids: &values,
                coefficients: &values,
            },
        )
        .unwrap();
    session.finish_interventions().unwrap();
    let step = session.take_step().unwrap();
    assert_eq!(step.step_usage.retained_bytes, 0);
    assert_eq!(step.step_usage.captures, 0);
    assert!(facts.calls.lock().unwrap().is_empty());
}

#[test]
fn shared_observer_attributes_routing_failures_and_missing_targets() {
    let (capture, plan) = routed(false);
    let mut session = CaptureSession::new(capture);
    session
        .enable_interventions(plan, Arc::new(Facts::new(1)))
        .unwrap();
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    {
        use crate::ActivationObserver;
        let mut observer = CaptureObserver::new(&mut session, Backend::default(), |error| error);
        assert!(observer.finish().is_err());
        observer.routing_control("router", 2).unwrap().unwrap();
        observer.routing_failed("router", "native selector failure");
        assert!(observer.finish().is_err());
    }
    assert!(matches!(
        session.take_step().unwrap().interventions[0].outcome,
        InterventionOutcome::Failed { .. }
    ));
}

#[test]
fn runtime_activation_limits_precede_evidence_and_native_patching() {
    let (capture, plan) = plans(
        vec![operation(
            "zero",
            InterventionAction::Zero {
                dtype: InterventionDtype::Float32,
            },
        )],
        false,
    );
    let mut facts = Facts::new(0);
    facts.max_index = 3;
    let mut session = CaptureSession::new(capture);
    session
        .enable_interventions(plan.clone(), Arc::new(facts))
        .unwrap();
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    let mut backend = Backend::default();
    assert!(session
        .intervene(
            &mut backend,
            &plan.points()[0].path,
            &Value {
                shape: vec![2, 2],
                data: vec![1.; 4]
            }
        )
        .is_err());
    assert_eq!(backend.applications, 0);
    assert_eq!(backend.copies, 0);
    assert!(matches!(
        session.take_step().unwrap().interventions[0].outcome,
        InterventionOutcome::Failed { .. }
    ));
}

#[test]
fn runtime_route_limits_and_missing_original_evidence_fail_explicitly() {
    let (capture, plan) = routed(true);
    let mut facts = Facts::new(1);
    facts.max_index = 3;
    let mut session = CaptureSession::new(capture.clone());
    session
        .enable_interventions(plan.clone(), Arc::new(facts))
        .unwrap();
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    assert!(session.routing_control("router", 2).is_err());
    assert!(matches!(
        session.take_step().unwrap().interventions[0].outcome,
        InterventionOutcome::Failed { .. }
    ));

    let mut session = CaptureSession::new(capture);
    session
        .enable_interventions(plan, Arc::new(Facts::new(1)))
        .unwrap();
    session.begin_step(CapturePhase::Prefill, 0).unwrap();
    session.routing_control("router", 2).unwrap().unwrap();
    let values = Value {
        shape: vec![2, 2],
        data: vec![1.; 4],
    };
    assert!(session
        .routing_applied(
            &mut Backend::default(),
            "router",
            None,
            crate::RoutingDecision {
                ids: &values,
                coefficients: &values
            }
        )
        .is_err());
    assert!(session.finish_interventions().is_err());
    assert!(matches!(
        session.take_step().unwrap().interventions[0].outcome,
        InterventionOutcome::Failed { .. }
    ));
}

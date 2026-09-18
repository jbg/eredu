mod speculative;
mod windows;
use super::*;
use crate::capture::CaptureInvocationSelection;
use std::{cell::Cell, sync::Arc};

fn bounds() -> CaptureInvocationBounds {
    CaptureInvocationBounds {
        batch: 1,
        max_sequence: 4,
        max_context: None,
        max_predictions: 6,
    }
}
fn setup(captures: u64, sliced: bool) -> (CaptureSession, CaptureDiscovery, InterventionDiscovery) {
    let (original, intervention) = plans(
        vec![operation(
            "edit",
            InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: 2.0,
            },
        )],
        true,
    );
    let discovery = session::discovery(&intervention);
    let catalog = ObservationCatalog {
        schema_version: 1,
        completeness: DescriptionCompleteness::Complete,
        points: discovery
            .points
            .iter()
            .map(InterventionPoint::observation_geometry)
            .collect(),
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: CaptureCapabilities {
            transformations: vec![CaptureTransformKind::Preview],
            ..Default::default()
        },
        points: vec![ObservationSupport {
            path: "block.output".into(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let mut raw = original.plan().clone();
    raw.selections[0].transform = CaptureTransform::Preview { max_elements: 64 };
    raw.limits.cumulative.captures = captures;
    let capture = raw
        .admit_invocations(&catalog, &support, &support.capture, bounds())
        .unwrap();
    let mut raw = intervention.plan().clone();
    raw.operations[0].evidence = InterventionEvidence::Preview { max_elements: 64 };
    if sliced {
        raw.operations[0].slices = vec![CaptureSlice {
            axis: "sequence".into(),
            start: 1,
            end: 3,
            stride: 1,
        }];
        raw.operations[0].action = InterventionAction::Mask {
            dtype: InterventionDtype::Float32,
            shape: vec![2, 2],
            keep: vec![true, false, true, false],
        };
    }
    let intervention = raw
        .admit_invocations(&discovery, bounds(), "session")
        .unwrap();
    preflight(&capture, &intervention, &Estimates).unwrap();
    let mut run = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(capture));
    run.enable_interventions(intervention, Arc::new(Estimates))
        .unwrap();
    (
        run,
        CaptureDiscovery {
            artifact_identity: "source".into(),
            catalog,
            support,
        },
        discovery,
    )
}
fn shape(sequence: u64) -> CaptureInvocationShape {
    CaptureInvocationShape {
        batch: 1,
        sequence,
        context: None,
    }
}
fn input(sequence: u64) -> Value {
    Value {
        shape: vec![sequence, 2],
        data: (1..=2 * sequence).map(|n| n as f32).collect(),
    }
}
fn execute(run: &mut CaptureSession, backend: &mut Backend, value: &Value) -> Value {
    run.observe(backend, "block.output", value).unwrap();
    let output = run
        .intervene(backend, "block.output", value)
        .unwrap()
        .unwrap();
    run.finish_interventions().unwrap();
    output
}

#[test]
fn independent_geometry_preserves_prediction_coordinates_and_causal_evidence() {
    let (mut run, discovery, _) = setup(64, false);
    let checkpoint = run.checkpoint(&discovery).unwrap();
    let mut backend = Backend::default();
    let mut spent = CaptureUsage::default();
    // Multi-row cached verification and one-row proposals may share a schedule
    // coordinate. A later seed also need not masquerade as prediction zero.
    for (phase, prediction, sequence) in [
        (CapturePhase::Decode, 0, 3),
        (CapturePhase::Decode, 0, 1),
        (CapturePhase::Prefill, 4, 2),
    ] {
        run.begin_invocation(
            phase,
            prediction,
            shape(sequence),
            CaptureInvocationSelection::default(),
        )
        .unwrap();
        let input = input(sequence);
        let output = execute(&mut run, &mut backend, &input);
        assert_eq!(
            output.data,
            input.data.iter().map(|x| 2.0 * x).collect::<Vec<_>>()
        );
        let record = run.take_step().unwrap();
        assert_eq!(record.phase, phase);
        assert_eq!(record.prediction_index, prediction);
        assert_eq!(record.invocation, Some(shape(sequence)));
        assert_eq!(values(&record.records[0]), input.data);
        assert_eq!(values(&record.interventions[0].evidence[0]), input.data);
        assert_eq!(values(&record.interventions[0].evidence[1]), output.data);
        assert!(record.cumulative_usage.host_bytes > spent.host_bytes);
        spent = record.cumulative_usage;
        assert_eq!(
            serde_json::from_value::<CapturedStep>(serde_json::to_value(&record).unwrap()).unwrap(),
            record
        );
        run.restore(&checkpoint).unwrap();
        assert_eq!(run.cumulative_usage(), spent);
    }
    assert_eq!(backend.applications, 3);
    assert_eq!(backend.copies, 9);
    assert!(run.begin_step(CapturePhase::Decode, 1).is_err());
}

#[test]
fn exact_position_masks_and_scope_applicability_precede_observed_work() {
    let (mut run, _, _) = setup(64, true);
    let mut backend = Backend::default();
    assert!(run
        .begin_invocation(
            CapturePhase::Decode,
            1,
            shape(1),
            CaptureInvocationSelection::default()
        )
        .is_err());
    assert_eq!(run.cumulative_usage(), CaptureUsage::default());
    assert_eq!((backend.copies, backend.applications), (0, 0));
    run.begin_invocation(
        CapturePhase::Decode,
        1,
        shape(3),
        CaptureInvocationSelection::default(),
    )
    .unwrap();
    let output = execute(&mut run, &mut backend, &input(3));
    assert_eq!(output.data, [1.0, 2.0, 3.0, 0.0, 5.0, 0.0]);
    run.take_step().unwrap();
    let spent = run.cumulative_usage();
    let disabled = [false];
    run.begin_invocation(
        CapturePhase::Decode,
        1,
        shape(1),
        CaptureInvocationSelection {
            captures: Some(&disabled),
            interventions: Some(&disabled),
        },
    )
    .unwrap();
    let source = input(1);
    run.observe(&mut backend, "block.output", &source).unwrap();
    assert!(run
        .intervene(&mut backend, "block.output", &source)
        .unwrap()
        .is_none());
    run.finish_interventions().unwrap();
    let record = run.take_step().unwrap();
    assert_eq!(
        record.records[0].outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::NotInvoked
        }
    );
    assert_eq!(
        record.interventions[0].outcome,
        InterventionOutcome::Inactive
    );
    assert!(record.interventions[0].evidence.iter().all(|e| e.outcome
        == CaptureOutcome::Skipped {
            reason: CaptureSkipReason::NotInvoked
        }));
    assert!(
        record.cumulative_usage.host_bytes > spent.host_bytes,
        "scope skips retain accounted envelopes"
    );
    assert_eq!((backend.copies, backend.applications), (3, 1));
    for bad in [
        CaptureInvocationShape {
            batch: 2,
            ..shape(1)
        },
        shape(5),
        CaptureInvocationShape {
            context: Some(7),
            ..shape(1)
        },
    ] {
        assert!(run
            .begin_invocation(
                CapturePhase::Decode,
                1,
                bad,
                CaptureInvocationSelection::default()
            )
            .is_err());
    }
}

#[test]
fn invocation_restore_never_refunds_generated_capture_or_intervention_work() {
    let (mut run, discovery, _) = setup(3, false);
    let saved = run.checkpoint(&discovery).unwrap();
    let mut backend = Backend::default();
    run.begin_invocation(
        CapturePhase::Decode,
        1,
        shape(2),
        CaptureInvocationSelection::default(),
    )
    .unwrap();
    execute(&mut run, &mut backend, &input(2));
    let spent = run.take_step().unwrap().cumulative_usage;
    assert_eq!(spent.captures, 3);
    run.restore(&saved).unwrap();
    run.begin_invocation(
        CapturePhase::Decode,
        1,
        shape(3),
        CaptureInvocationSelection::default(),
    )
    .unwrap();
    let generated = Cell::new(0);
    let value = input(3);
    let failure = run
        .observe_generated(
            &mut backend,
            "block.output",
            &value,
            &GeneratedCaptureSource {
                creation_bytes: 24,
                source_dtype: Some(eredu_core::checkpoint::TensorDtype::F32),
            },
            &mut || {
                generated.set(generated.get() + 1);
                Ok::<_, CaptureExecutionError<std::io::Error>>(value.clone())
            },
            &|error| error,
        )
        .unwrap_err();
    assert!(matches!(
        failure,
        CaptureExecutionError::Admission(CaptureError::Limit {
            budget: CaptureBudget::Captures,
            cumulative: true
        })
    ));
    assert_eq!(generated.get(), 0);
    let failed = run.take_step().unwrap();
    assert!(matches!(
        failed.records[0].outcome,
        CaptureOutcome::Failed {
            reason: CaptureFailureReason::Limit { .. },
            ..
        }
    ));
    assert_eq!(failed.cumulative_usage.captures, spent.captures);
    assert!(failed.cumulative_usage.host_bytes > spent.host_bytes);
    run.restore(&saved).unwrap();
    assert_eq!(run.cumulative_usage(), failed.cumulative_usage);
    assert_eq!((backend.copies, backend.applications), (3, 1));
}

#[test]
fn invocation_authority_is_bound_to_shape_mode_and_survives_child_readmission() {
    let (mut run, discovery, intervention_discovery) = setup(64, false);
    let none = CapturePlan::none()
        .admit_invocations(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            bounds(),
        )
        .unwrap();
    crate::capture::preflight(&none, estimate).unwrap();
    let mut absent = None;
    install_session(&mut absent, none, None).unwrap();
    assert!(
        absent.is_none(),
        "capture-none retains the uninstrumented path"
    );
    let admitted = run.plan().clone();
    let ordinary = admitted
        .plan()
        .clone()
        .admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            admitted.request(),
        )
        .unwrap();
    assert_ne!(ordinary.identity(), admitted.identity());
    let intervention = run.intervention_plan().unwrap();
    let ordinary_intervention = intervention
        .plan()
        .clone()
        .admit(&intervention_discovery, ordinary.request(), "session")
        .unwrap();
    assert_ne!(
        ordinary_intervention.intent_identity(),
        intervention.intent_identity()
    );
    assert!(intervention
        .validate_actual(
            0,
            CapturePhase::Decode,
            1,
            &[1, 2],
            Some(InterventionDtype::Float32)
        )
        .is_err());
    assert!(CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(ordinary))
        .enable_interventions(intervention.clone(), Arc::new(Estimates))
        .is_err());
    let mut backend = Backend::default();
    run.begin_invocation(
        CapturePhase::Decode,
        1,
        shape(2),
        CaptureInvocationSelection::default(),
    )
    .unwrap();
    execute(&mut run, &mut backend, &input(2));
    run.take_step().unwrap();
    let saved = run.checkpoint(&discovery).unwrap();
    let mut child = saved
        .fork(
            crate::capture::CaptureForkRequest {
                discovery: &discovery,
                max_predictions: 6,
                limits: admitted.plan().limits.clone(),
                intervention: Some(crate::capture::InterventionForkRequest {
                    discovery: &intervention_discovery,
                    session_id: "child",
                    replacement: None,
                    estimator: Arc::new(Estimates),
                }),
            },
            estimate,
        )
        .unwrap();
    assert_eq!(child.plan().invocation_bounds(), Some(bounds()));
    assert_eq!(child.cumulative_usage(), run.cumulative_usage());
    child
        .begin_invocation(
            CapturePhase::Decode,
            2,
            shape(3),
            CaptureInvocationSelection::default(),
        )
        .unwrap();
    execute(&mut child, &mut backend, &input(3));
    let record = child.take_step().unwrap();
    assert_eq!(record.invocation, Some(shape(3)));
    assert!(record.cumulative_usage.captures > run.cumulative_usage().captures);
}

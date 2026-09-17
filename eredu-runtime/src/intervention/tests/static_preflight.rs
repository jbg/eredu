use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
struct Retires(Arc<AtomicBool>);
impl Drop for Retires {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}
fn bounded(
    source: &AdmittedCapturePlan,
    retained: u64,
    policy: CaptureLimitPolicy,
) -> AdmittedCapturePlan {
    let mut plan = source.plan().clone();
    plan.limits.per_step.retained_bytes = retained;
    plan.limits.cumulative.retained_bytes = retained;
    plan.limits.on_limit = policy;
    let capabilities = CaptureCapabilities {
        transformations: vec![CaptureTransformKind::Preview],
        ..Default::default()
    };
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: source.points().to_vec(),
        completeness: DescriptionCompleteness::Complete,
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: capabilities.clone(),
        points: source
            .points()
            .iter()
            .map(|point| ObservationSupport {
                path: point.path.clone(),
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            })
            .collect(),
    };
    plan.admit(&catalog, &support, &capabilities, source.request())
        .unwrap()
}
#[test]
fn prepared_static_preflight_matches_shared_budget_and_retains_scratch() {
    let mut edit = operation(
        "scale-strided",
        InterventionAction::Scale {
            dtype: InterventionDtype::Float32,
            factor: 1.5,
        },
    );
    edit.evidence = InterventionEvidence::None;
    edit.slices = vec![CaptureSlice {
        axis: "hidden".into(),
        start: 0,
        end: 2,
        stride: 2,
    }];
    let retired = Arc::new(AtomicBool::new(false));
    let mut scratch = StaticInterventionPreflight::prepare(HostPreparationAuthority::retain(
        Retires(retired.clone()),
    ))
    .unwrap();
    for observe in [false, true] {
        let (capture, intervention) = plans(vec![edit.clone()], observe);
        for policy in [CaptureLimitPolicy::Fail, CaptureLimitPolicy::Skip] {
            for limit in [0, 63, 64, 127, 128, 255, 512, 1024] {
                let capture = bounded(&capture, limit, policy);
                let expected = preflight(&capture, &intervention, &Estimates);
                let actual = scratch.run(&capture, &intervention, &Estimates);
                match (expected, actual) {
                    (Ok(()), Ok(())) => (),
                    (
                        Err(CaptureError::Limit {
                            budget: left,
                            cumulative: lc,
                        }),
                        Err(StaticInterventionPreflightError::Capture(CaptureError::Limit {
                            budget: right,
                            cumulative: rc,
                        })),
                    ) => {
                        assert_eq!(left, right);
                        assert_eq!(lc, rc);
                    }
                    (left, right) => panic!("preflight differs: {left:?} / {right:?}"),
                }
            }
        }
    }
    assert!(!retired.load(Ordering::SeqCst));
    drop(scratch);
    assert!(retired.load(Ordering::SeqCst));
}

#[test]
fn prepared_evidence_preflight_uses_source_companions_and_enclosing_limits() {
    let pool = crate::working_memory::WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let retired = Arc::new(AtomicBool::new(false));
    let mut scratch = StaticInterventionPreflight::prepare(HostPreparationAuthority::retain(
        Retires(retired.clone()),
    ))
    .unwrap();
    for evidence in [
        InterventionEvidence::Preview { max_elements: 2 },
        InterventionEvidence::Summary,
    ] {
        let mut edit = operation(
            "selected-scale",
            InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: 1.5,
            },
        );
        edit.evidence = evidence;
        edit.slices = vec![CaptureSlice {
            axis: "hidden".into(),
            start: 0,
            end: 2,
            stride: 2,
        }];
        for ordinary in [false, true] {
            let (capture, intervention) = plans(vec![edit.clone()], ordinary);
            assert!(matches!(
                scratch.run(&capture, &intervention, &Estimates),
                Err(StaticInterventionPreflightError::Profile)
            ));
            let source = pool
                .compile_intervention_source(
                    PreparedInterventionPlanCopy::inspect(&intervention).unwrap(),
                )
                .unwrap();
            assert_eq!(
                source
                    .plan()
                    .evidence(0)
                    .unwrap()
                    .geometry_source()
                    .plan()
                    .limits,
                CapturePlan::none().limits
            );
            for policy in [CaptureLimitPolicy::Fail, CaptureLimitPolicy::Skip] {
                for limit in [0, 127, 255, 511, 1024, 2048] {
                    let capture = bounded(&capture, limit, policy);
                    let expected = preflight(&capture, &intervention, &Estimates);
                    let actual = scratch.run_with_source(&capture, source.plan(), &Estimates);
                    match (expected, actual) {
                        (Ok(()), Ok(())) => (),
                        (
                            Err(CaptureError::Limit {
                                budget: left,
                                cumulative: lc,
                            }),
                            Err(StaticInterventionPreflightError::Capture(CaptureError::Limit {
                                budget: right,
                                cumulative: rc,
                            })),
                        ) => {
                            assert_eq!(left, right);
                            assert_eq!(lc, rc);
                        }
                        (left, right) => panic!("evidence preflight differs: {left:?} / {right:?}"),
                    }
                }
            }
            drop(source);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
    assert!(!retired.load(Ordering::SeqCst));
    drop(scratch);
    assert!(retired.load(Ordering::SeqCst));
}


#[test]
fn prepared_invocation_readmission_matches_ordinary_current_usage_and_evidence() {
    let pool = crate::working_memory::WorkingMemoryPool::new(1 << 25, 0).unwrap();
    let retired = Arc::new(AtomicBool::new(false));
    let mut scratch = StaticInterventionPreflight::prepare(HostPreparationAuthority::retain(Retires(retired.clone()))).unwrap();
    let bounds = CaptureInvocationBounds { batch: 1, max_sequence: 5, max_context: None, max_predictions: 7 };
    let mut successes = 0;
    let mut refusals = 0;
    for evidence in [InterventionEvidence::None, InterventionEvidence::Preview { max_elements: 3 }, InterventionEvidence::Summary] {
        let mut edit = operation("nonzero-strided", InterventionAction::Scale { dtype: InterventionDtype::Float32, factor: -1.25 });
        edit.evidence = evidence;
        edit.slices = vec![CaptureSlice { axis: "hidden".into(), start: 0, end: 2, stride: 2 }];
        let (ordinary, edit) = plans(vec![edit], true);
        let discovery = session::discovery(&edit);
        let mut raw_edit = edit.plan().clone();
        raw_edit.operations[0].slices.insert(0, CaptureSlice { axis: "sequence".into(), start: 1, end: 5, stride: 2 });
        let edit = raw_edit.admit_invocations(&discovery, bounds, "session").unwrap();
        let source = pool.compile_intervention_source(PreparedInterventionPlanCopy::inspect(&edit).unwrap()).unwrap();
        let capabilities = CaptureCapabilities { transformations: vec![CaptureTransformKind::Preview], ..Default::default() };
        let catalog = ObservationCatalog { schema_version: 1, completeness: DescriptionCompleteness::Complete, points: ordinary.points().to_vec() };
        let support = ObservationSupportReport { schema_version: 1, capture: capabilities.clone(), points: vec![ObservationSupport {
            path: "block.output".into(), prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported, floating_to_f32: true,
        }] };
        for policy in [CaptureLimitPolicy::Fail, CaptureLimitPolicy::Skip] {
            for limit in [0, 127, 255, 511, 1024, 2048] {
                let mut raw = ordinary.plan().clone();
                raw.limits.per_step.retained_bytes = limit;
                raw.limits.cumulative.retained_bytes = limit;
                raw.limits.on_limit = policy;
                let capture = raw.admit_invocations(&catalog, &support, &capabilities, bounds).unwrap();
                for inherited in [CaptureUsage::default(), CaptureUsage { retained_bytes: 333, ..Default::default() }] {
                    let expected = preflight_continuation(&capture, &edit, &Estimates, 0, inherited);
                    let actual = scratch.run_invocation_with_source(&capture, source.plan(), &Estimates, inherited);
                    match (expected, actual) {
                        (Ok(()), Ok(())) => successes += 1,
                        (Err(CaptureError::Limit { budget: left, cumulative: lc }),
                            Err(StaticInterventionPreflightError::Capture(CaptureError::Limit { budget: right, cumulative: rc }))) => {
                            assert_eq!((left, lc), (right, rc)); refusals += 1;
                        }
                        (left, right) => panic!("invocation continuation differs: {left:?} / {right:?}"),
                    }
                }
            }
        }
        assert!(matches!(scratch.run_invocation_with_source(&ordinary, source.plan(), &Estimates, CaptureUsage::default()),
            Err(StaticInterventionPreflightError::Request)));
    }
    assert!(successes > 0 && refusals > 0);
    assert!(!retired.load(Ordering::SeqCst));
    drop(scratch);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

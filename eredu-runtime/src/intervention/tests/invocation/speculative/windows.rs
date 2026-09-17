use super::*;
use eredu_core::speculative::{
    SpeculativePrefillReductionGeometry, SpeculativePrefillReductionStatus as Status,
    SpeculativePrefillSpan,
};

fn run_with_evidence(evidence: InterventionEvidence, stride: u64) -> CaptureSession {
    run_with_limit(evidence, stride, None)
}
fn run_with_limit(
    evidence: InterventionEvidence,
    stride: u64,
    limit: Option<(bool, u64)>,
) -> CaptureSession {
    let (original, mut discovery, edits) = setup(64, false);
    let mut captures = original.plan().plan().clone();
    captures.selections[0].transform = match evidence {
        InterventionEvidence::Summary => CaptureTransform::Summary,
        _ => CaptureTransform::Preview { max_elements: 3 },
    };
    if let Some((host, n)) = limit {
        if host {
            captures.limits.per_step.host_bytes = n;
        } else {
            captures.limits.per_step.encoded_bytes = n;
        }
    }
    discovery
        .support
        .capture
        .transformations
        .push(CaptureTransformKind::Summary);
    let capture = captures
        .admit_invocations(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            bounds(),
        )
        .unwrap();
    let mut operations = original.intervention_plan().unwrap().plan().clone();
    operations.operations[0].slices = vec![CaptureSlice {
        axis: "sequence".into(),
        start: 1,
        end: 3,
        stride,
    }];
    operations.operations[0].evidence = evidence.clone();
    let mut second = operations.operations[0].clone();
    second.id = "second".into();
    second.action = InterventionAction::Scale {
        dtype: InterventionDtype::Float32,
        factor: -0.5,
    };
    operations.operations.push(second);
    let mut run = CaptureSession::new(capture);
    run.enable_interventions(
        operations
            .admit_invocations(&edits, bounds(), "session")
            .unwrap(),
        Arc::new(Estimates),
    )
    .unwrap();
    run
}
fn observer_for(run: CaptureSession, scope: Scope) -> Observer {
    observer_with_failure(run, scope, false)
}
fn observer_with_failure(run: CaptureSession, scope: Scope, fail: bool) -> Observer {
    SpeculativeCaptureObserver::new(
        run,
        Provider { fail },
        map as fn(&CaptureExecutionError<std::io::Error>) -> String,
        SpeculativeRequestId::new(0),
        vec![scope],
        vec![scope, scope],
    )
    .unwrap()
}
fn start(observer: &mut Observer) {
    observer.set_activation_origin(Some(origin(0)));
    observer.set_prefill_reduction_geometry(SpeculativePrefillReductionGeometry {
        target_sequence: 4,
        prediction_sequence: 3,
    });
}
fn window(
    observer: &mut Observer,
    phase: Phase,
    a: u64,
    b: u64,
    transactional: bool,
) -> Result<Value, String> {
    let seed = phase == Phase::PredictionPrefill;
    // These are real physical callback coordinates: the seed skips the first
    // prompt token while its retained hidden rows begin at logical zero.
    let token_start = a + u64::from(seed);
    observer.set_prefill_span(Some(SpeculativePrefillSpan {
        prompt_tokens: 4,
        input_start: if seed && a == 0 { 0 } else { token_start },
        input_end: b + u64::from(seed),
        position: if seed && a == 0 { 0 } else { token_start },
        hidden_start: a,
        token_start,
        sequence: b - a,
        seed_start: a,
    }));
    let value = Value {
        shape: vec![b - a, 2],
        data: (a * 2 + 1..=b * 2)
            .map(|v| v as f32 + if seed { 10. } else { 0. })
            .collect(),
    };
    let result = with_speculative_activation(Some(observer), phase, (b - a) as usize, |o| {
        let o = o.unwrap();
        let epoch = DistributedCommitEpoch::new(a + 1).unwrap();
        if transactional {
            o.prepare_transaction(epoch, crate::ExpertPass::Prefill)?;
        }
        let value = crate::observe_and_intervene(o, "block.output", &value)?;
        if transactional {
            o.finish()?;
            o.complete_transaction(epoch)?;
            o.finish_transaction(epoch, true);
        }
        Ok(value)
    });
    observer.set_prefill_span(None);
    result
}
fn assert_payload(a: &CaptureRecord, b: &CaptureRecord) {
    assert_eq!(a.source_shape, b.source_shape);
    assert_eq!(a.selected_shape, b.selected_shape);
    assert_eq!(a.source_dtype, b.source_dtype);
    assert_eq!(a.outcome, b.outcome);
    match (&a.payload, &b.payload) {
        (Some(CapturePayload::Summary(a)), Some(CapturePayload::Summary(b))) => {
            assert_eq!(
                (a.elements, a.finite, a.min, a.max),
                (b.elements, b.finite, b.min, b.max)
            );
            assert!((a.mean.unwrap() - b.mean.unwrap()).abs() < 1e-12);
            assert!((a.rms.unwrap() - b.rms.unwrap()).abs() < 1e-12);
        }
        _ => assert_eq!(a.payload, b.payload),
    }
}
#[test]
fn split_dense_evidence_matches_ordinary_ordered_edits_and_shifted_seed() {
    for evidence in [
        InterventionEvidence::Preview { max_elements: 3 },
        InterventionEvidence::Summary,
    ] {
        for stride in [1, 2] {
            for seed in [false, true] {
                for transactional in [false, true] {
                    let phase = if seed {
                        Phase::PredictionPrefill
                    } else {
                        Phase::TargetPrefill
                    };
                    let scope = if seed {
                        Scope::Prediction { depth: 0 }
                    } else {
                        Scope::Target
                    };
                    let rows = if seed { 3 } else { 4 };
                    let mut baseline = run_with_evidence(evidence.clone(), stride);
                    baseline
                        .begin_invocation(
                            CapturePhase::Prefill,
                            0,
                            shape(rows),
                            CaptureInvocationSelection::default(),
                        )
                        .unwrap();
                    let value = Value {
                        shape: vec![rows, 2],
                        data: (1..=rows * 2)
                            .map(|v| v as f32 + if seed { 10. } else { 0. })
                            .collect(),
                    };
                    let expected = execute(&mut baseline, &mut Backend::default(), &value);
                    let expected_step = baseline.take_step().unwrap();
                    let mut observer =
                        observer_for(run_with_evidence(evidence.clone(), stride), scope);
                    start(&mut observer);
                    let mut actual = Vec::new();
                    for (a, b) in [(0, 1), (1, 3), (3, rows)]
                        .into_iter()
                        .filter(|(a, b)| a < b)
                    {
                        actual.extend(
                            window(&mut observer, phase, a, b, transactional)
                                .unwrap()
                                .data,
                        );
                    }
                    observer.complete_prefill_reductions().unwrap();
                    observer.finish_prefill_reductions(true);
                    assert_eq!(actual, expected.data);
                    let steps: Vec<_> =
                        std::iter::from_fn(|| observer.take_activation_capture()).collect();
                    assert!(steps.iter().all(|s| s.completed));
                    let report = steps.last().unwrap().prefill_reductions.as_ref().unwrap();
                    assert_eq!(report.interventions.len(), 2);
                    for (entry, expected) in report
                        .interventions
                        .iter()
                        .zip(&expected_step.interventions)
                    {
                        assert_eq!(entry.status, Status::Complete);
                        assert_eq!(entry.covered_sequence, rows);
                        assert_eq!(entry.windows, steps.len() as u64);
                        assert_eq!(entry.record.outcome, expected.outcome);
                        for (a, b) in entry.record.evidence.iter().zip(&expected.evidence) {
                            assert_payload(a, b);
                        }
                    }
                    // Earlier physical envelopes remain independently inspectable.
                    assert!(steps[0].prefill_reductions.is_none());
                    // The pre-existing dense worker reports successful application
                    // even when the selected intersection has no replacement. Its
                    // evidence must still prove that this physical slice is empty.
                    for record in &steps[0].captures.as_step().interventions {
                        assert_eq!(record.outcome, InterventionOutcome::Applied);
                        for evidence in &record.evidence {
                            assert_eq!(
                                evidence.outcome,
                                CaptureOutcome::Skipped {
                                    reason: CaptureSkipReason::NotInvoked
                                }
                            );
                            assert!(evidence.payload.is_none());
                        }
                    }
                    let json = serde_json::to_value(report).unwrap();
                    assert_eq!(
                        serde_json::from_value::<
                            eredu_core::speculative::SpeculativePrefillReductions,
                        >(json)
                        .unwrap(),
                        *report.as_reductions()
                    );
                }
            }
        }
    }
}
#[test]
fn split_evidence_abort_and_failed_edit_preserve_physical_prefix_without_refund() {
    for fail in [false, true] {
        let mut observer = observer_with_failure(
            run_with_evidence(InterventionEvidence::Preview { max_elements: 3 }, 1),
            Scope::Target,
            fail,
        );
        start(&mut observer);
        window(&mut observer, Phase::TargetPrefill, 0, 1, true).unwrap();
        if fail {
            assert!(window(&mut observer, Phase::TargetPrefill, 1, 3, true).is_err());
        } else {
            assert!(observer.complete_prefill_reductions().is_err());
        }
        let usage = observer.session().cumulative_usage();
        observer.finish_prefill_reductions(false);
        assert_eq!(observer.session().cumulative_usage(), usage);
        let steps: Vec<_> = std::iter::from_fn(|| observer.take_activation_capture()).collect();
        assert!(steps[0].completed);
        assert_eq!(steps[0].captures.as_step().outcome, CaptureStepOutcome::Committed);
        let report = steps.last().unwrap().prefill_reductions.as_ref().unwrap();
        assert!(report
            .interventions
            .iter()
            .all(|entry| entry.status != Status::Complete
                && entry.record.evidence.iter().all(|e| e.payload.is_none())));
        if fail {
            assert!(!steps.last().unwrap().completed);
            assert!(steps.last().unwrap().captures.as_step().interventions[0].evidence[0]
                .payload
                .is_some());
            assert!(matches!(
                observer.take_activation_error(),
                Some(SpeculativeControlError::Backend(_))
            ));
        }
    }
}

#[test]
fn aggregate_metadata_exact_and_one_short_reject_before_model_callback() {
    let span = SpeculativePrefillSpan {
        prompt_tokens: 4,
        input_start: 0,
        input_end: 1,
        position: 0,
        hidden_start: 0,
        token_start: 0,
        sequence: 1,
        seed_start: 0,
    };
    let mut probe = observer_for(
        run_with_evidence(InterventionEvidence::Summary, 1),
        Scope::Target,
    );
    start(&mut probe);
    probe.set_prefill_span(Some(span));
    probe
        .begin_activation_invocation(Phase::TargetPrefill, 1)
        .unwrap();
    let usage = probe.session().cumulative_usage();
    probe.finish_activation_invocation(false);
    probe.finish_prefill_reductions(false);
    for (host, n) in [(true, usage.host_bytes), (false, usage.encoded_bytes)] {
        for short in [false, true] {
            let mut observer = observer_for(
                run_with_limit(
                    InterventionEvidence::Summary,
                    1,
                    Some((host, n - u64::from(short))),
                ),
                Scope::Target,
            );
            start(&mut observer);
            observer.set_prefill_span(Some(span));
            let result = observer.begin_activation_invocation(Phase::TargetPrefill, 1);
            assert_eq!(result.is_ok(), !short);
            if short {
                assert!(matches!(
                    observer.take_activation_error(),
                    Some(SpeculativeControlError::Capture(CaptureError::Limit { .. }))
                ));
            }
            let retained = observer.session().cumulative_usage();
            assert!(retained.host_bytes > 0 && retained.encoded_bytes > 0);
            if short {
                assert!(if host {
                    retained.host_bytes < n
                } else {
                    retained.encoded_bytes < n
                });
            }
            observer.finish_activation_invocation(false);
            observer.finish_prefill_reductions(false);
            assert_eq!(observer.session().cumulative_usage(), retained);
        }
    }
}

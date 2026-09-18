//! The actual capture/edit driver consumes global selection through physical spans.
use super::*;

fn window_run(limit: u64) -> (CaptureSession, CaptureDiscovery) {
    window_run_shape(limit, 5)
}
fn window_run_shape(limit: u64, total: u64) -> (CaptureSession, CaptureDiscovery) {
    let (original, mut discovery, edits) = setup(64, false);
    let bounds = CaptureInvocationBounds {
        max_sequence: 5,
        ..bounds()
    };
    discovery.support.capture.transformations.extend([
        CaptureTransformKind::Slice,
        CaptureTransformKind::FullTensor,
    ]);
    let mut capture = original.plan().plan().clone();
    capture.selections[0].transform = CaptureTransform::Slice;
    capture.selections[0].slices = vec![CaptureSlice {
        axis: "sequence".into(),
        start: 1,
        end: total,
        stride: 2,
    }];
    capture.limits.cumulative.captures = limit;
    let capture = capture
        .admit_invocations(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            bounds,
        )
        .unwrap();
    let mut edit = original.intervention_plan().unwrap().plan().clone();
    edit.operations[0].slices = capture.plan().selections[0].slices.clone();
    edit.operations[0].evidence = InterventionEvidence::None;
    edit.operations[0].action = InterventionAction::Replace {
        tensor: InterventionTensor {
            shape: vec![2, 2],
            values: InterventionValues::Float32(vec![31.0, 32.0, 71.0, 72.0]),
        },
    };
    let edit = edit.admit_invocations(&edits, bounds, "session").unwrap();
    let mut session = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(capture));
    session
        .enable_interventions(edit, Arc::new(Estimates))
        .unwrap();
    (session, discovery)
}

#[test]
fn window_stride_and_replacement_payload_match_unsplit_global_selection() {
    let (mut ordinary, _) = window_run(64);
    ordinary
        .begin_invocation(
            CapturePhase::Prefill,
            0,
            shape(5),
            CaptureInvocationSelection::default(),
        )
        .unwrap();
    let input = input(5);
    let mut reference = Backend::default();
    let expected = execute(&mut ordinary, &mut reference, &input);
    let full = ordinary.take_step().unwrap();
    let (mut spans, discovery) = window_run(64);
    let saved = spans.checkpoint(&discovery).unwrap();
    let mut backend = Backend::default();
    let mut observed = Vec::<f32>::new();
    let mut actual = Vec::new();
    let mut previous = CaptureUsage::default();
    for (start, end) in [(0, 2), (2, 4), (4, 5)] {
        spans
            .begin_invocation_window(
                CapturePhase::Prefill,
                0,
                shape(end - start),
                CaptureInvocationSelection::default(),
                CaptureInvocationWindow {
                    logical_sequence: 5,
                    start,
                },
            )
            .unwrap();
        let local = Value {
            shape: vec![end - start, 2],
            data: input.data[(start * 2) as usize..(end * 2) as usize].to_vec(),
        };
        spans.observe(&mut backend, "block.output", &local).unwrap();
        let edited = spans
            .intervene(&mut backend, "block.output", &local)
            .unwrap()
            .unwrap_or_else(|| local.clone());
        spans.finish_interventions().unwrap();
        actual.extend(edited.data);
        let step = spans.take_step().unwrap();
        if step.records[0].payload.is_some() {
            observed.extend(values(&step.records[0]));
        } else {
            assert_eq!(
                step.records[0].outcome,
                CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::NotInvoked
                }
            );
        }
        assert_eq!(step.prediction_index, 0);
        assert!(step.cumulative_usage.host_bytes > previous.host_bytes);
        previous = step.cumulative_usage;
    }
    assert_eq!(actual, expected.data);
    assert_eq!(observed, values(&full.records[0]));
    assert_eq!(observed, [3.0, 4.0, 7.0, 8.0]);
    assert_eq!(backend.applications, 2);
    assert_eq!(backend.copies, 2);
    spans.restore(&saved).unwrap();
    assert_eq!(
        spans.cumulative_usage(),
        previous,
        "restore cannot refund completed windows"
    );
}

#[test]
fn shifted_seed_windows_keep_the_hidden_row_origin_and_no_overlap_skips_copy() {
    let (mut run, _) = window_run_shape(64, 4);
    let mut backend = Backend::default();
    let mut captured = Vec::<f32>::new();
    // Five target rows seed four (hidden[i], token[i+1]) pairs in 1/2/1.
    for (start, end) in [(0, 1), (1, 3), (3, 4)] {
        // Exact four-row seed window, independent of the five-row target input.
        run.begin_invocation_window(
            CapturePhase::Prefill,
            0,
            shape(end - start),
            CaptureInvocationSelection {
                captures: None,
                interventions: Some(&[false]),
            },
            CaptureInvocationWindow {
                logical_sequence: 4,
                start,
            },
        )
        .unwrap();
        let local = Value {
            shape: vec![end - start, 2],
            data: (start * 2 + 1..=end * 2).map(|n| n as f32).collect(),
        };
        run.observe(&mut backend, "block.output", &local).unwrap();
        let step = run.take_step().unwrap();
        if step.records[0].payload.is_some() {
            captured.extend(values(&step.records[0]));
        }
    }
    assert_eq!(captured, [3.0, 4.0, 7.0, 8.0]);
    assert_eq!(backend.copies, 2);
}

#[test]
fn window_quota_failure_precedes_native_copy_and_whole_reduction_is_explicit() {
    let (mut run, _) = window_run(1);
    let mut backend = Backend::default();
    for (index, start) in [0, 2].into_iter().enumerate() {
        run.begin_invocation_window(
            CapturePhase::Prefill,
            0,
            shape(2),
            CaptureInvocationSelection {
                captures: None,
                interventions: Some(&[false]),
            },
            CaptureInvocationWindow {
                logical_sequence: 5,
                start,
            },
        )
        .unwrap();
        let result = run.observe(&mut backend, "block.output", &input(2));
        if index == 0 {
            result.unwrap();
        } else {
            assert!(matches!(
                result,
                Err(CaptureExecutionError::Admission(CaptureError::Limit {
                    cumulative: true,
                    ..
                }))
            ));
        }
        run.take_step().unwrap();
    }
    assert_eq!(backend.copies, 1);
    let (mut unsupported, _, _) = setup(64, false);
    assert!(matches!(
        unsupported.begin_invocation_window(
            CapturePhase::Prefill,
            0,
            shape(2),
            CaptureInvocationSelection::default(),
            CaptureInvocationWindow {
                logical_sequence: 4,
                start: 0
            }
        ),
        Err(CaptureError::Unsupported(_))
    ));
}

#[test]
fn invalid_window_row_declarations_reject_before_step_or_native_work() {
    for malformed in 0..3 {
        let (original, mut discovery, _) = setup(64, false);
        discovery
            .support
            .capture
            .transformations
            .push(CaptureTransformKind::FullTensor);
        let point = &mut discovery.catalog.points[0];
        match malformed {
            0 => point.axes = None,
            1 => point.axes.as_mut().unwrap()[0].dimension = SymbolicDimension::Known(2),
            2 => point.axes.as_mut().unwrap()[1].dimension = SymbolicDimension::TokenRows,
            _ => unreachable!(),
        }
        let mut plan = original.plan().plan().clone();
        plan.selections[0].transform = CaptureTransform::FullTensor;
        plan.selections[0].slices.clear();
        let plan = plan
            .admit_invocations(
                &discovery.catalog,
                &discovery.support,
                &discovery.support.capture,
                bounds(),
            )
            .unwrap();
        let mut run = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(plan));
        let before = run.cumulative_usage();
        let mut backend = Backend::default();
        let started = run.begin_invocation_window(
            CapturePhase::Prefill,
            0,
            shape(2),
            CaptureInvocationSelection::default(),
            CaptureInvocationWindow {
                logical_sequence: 4,
                start: 0,
            },
        );
        // A native caller must never reach observation after a failed preflight.
        if started.is_ok() {
            run.observe(&mut backend, "block.output", &input(2))
                .unwrap();
        }
        assert!(matches!(started, Err(CaptureError::Unsupported(_))));
        assert_eq!(run.cumulative_usage(), before);
        assert!(run.take_step().is_none());
        assert_eq!(backend.copies, 0);
        assert_eq!(backend.applications, 0);
    }
}

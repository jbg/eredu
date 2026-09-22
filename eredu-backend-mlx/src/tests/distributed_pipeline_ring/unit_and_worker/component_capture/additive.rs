// Public admitted edits at summed writes, compared with ordinary complete writes.
fn verify_additive_component_interventions(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    reference: &mut ModelRuntime<MlxBackend<'_>>,
    capture: &eredu_core::capture::AdmittedCapturePlan,
    sampling: eredu_core::ResolvedGenerationConfig,
    targets: &[String],
) {
    use eredu_core::{capture::*, intervention::*, TextGenerationBackend as _};
    if targets.is_empty() {
        return;
    }
    let discovery = MlxBackend::intervention_discovery(runtime).unwrap();
    let capture_discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let mut observations = capture.plan().clone();
    observations.selections.retain(|selection| {
        selection.path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH
            || targets.iter().any(|target| {
                selection.path == *target || selection.path == format!("{target}.effective")
            })
    });
    let capture = observations
        .admit(
            &capture_discovery.catalog,
            &capture_discovery.support,
            &capture_discovery.support.capture,
            capture.request(),
        )
        .unwrap();
    for action_index in 0..5 {
        let mut operations = Vec::new();
        for (index, target) in targets.iter().enumerate() {
            let point = discovery
                .points
                .iter()
                .find(|point| point.path == *target)
                .unwrap();
            let eredu_core::SymbolicDimension::Known(width) = point.axes.last().unwrap().dimension
            else {
                panic!("known write width")
            };
            let selected = width / 2;
            let shape = vec![1, 1, selected as u64];
            for (phase, position) in [(CapturePhase::Prefill, 1), (CapturePhase::Decode, 0)] {
                let payload = || InterventionTensor {
                    shape: shape.clone(),
                    values: InterventionValues::Float32(
                        (0..selected).map(|i| 0.03125 * (i as f32 + 1.0)).collect(),
                    ),
                };
                let dtype = InterventionDtype::Float32;
                let action = match action_index {
                    0 => InterventionAction::Zero { dtype },
                    1 => InterventionAction::Scale {
                        dtype,
                        factor: -0.5,
                    },
                    2 => InterventionAction::Add { tensor: payload() },
                    3 => InterventionAction::Replace { tensor: payload() },
                    _ => InterventionAction::Mask {
                        dtype,
                        shape: shape.clone(),
                        keep: (0..selected).map(|i| i % 2 == 0).collect(),
                    },
                };
                operations.push(InterventionOperation {
                    id: format!("additive-{index}-{phase:?}"),
                    target: target.clone(),
                    schedule: CaptureSchedule {
                        prefill: phase == CapturePhase::Prefill,
                        decode: phase == CapturePhase::Decode,
                        ..Default::default()
                    },
                    slices: vec![
                        CaptureSlice {
                            axis: "sequence".into(),
                            start: position,
                            end: position + 1,
                            stride: 1,
                        },
                        CaptureSlice {
                            axis: point.axes.last().unwrap().name.clone(),
                            start: 1,
                            end: width as u64,
                            stride: 2,
                        },
                    ],
                    action,
                    evidence: InterventionEvidence::Preview { max_elements: 128 },
                });
            }
        }
        let plan = InterventionPlan {
            schema_version: INTERVENTION_SCHEMA_VERSION,
            operations,
        };
        let run = |runtime: &mut ModelRuntime<MlxBackend<'_>>| {
            runtime.reset().unwrap();
            let admitted = plan
                .clone()
                .admit(
                    &MlxBackend::intervention_discovery(runtime).unwrap(),
                    capture.request(),
                    "native-additive-trial",
                )
                .unwrap();
            let interventions = runtime
                .backend()
                .memory_ledger()
                .compile_intervention_source(
                    eredu_core::intervention::PreparedInterventionPlanCopy::inspect(&admitted)
                        .unwrap(),
                )
                .unwrap()
                .plan()
                .clone();
            let mut generation = component_capture_generation(
                runtime,
                sampling,
                Some(eredu_core::TextPreparationOptions {
                    capture: Some(eredu_core::capture::SharedCapturePlan::new(capture.clone())),
                    interventions: Some(interventions),
                }),
            )
            .unwrap();
            (0..3)
                .map(|_| {
                    let token = generation.next().unwrap().unwrap().token_id();
                    (token, generation.take_captured_delivery().unwrap().unwrap())
                })
                .collect::<Vec<_>>()
        };
        let expected = run(reference);
        let actual = run(runtime);
        let compare = |a: &CaptureRecord, b: &CaptureRecord| {
            assert_eq!(
                (&a.path, &a.outcome, &a.position),
                (&b.path, &b.outcome, &b.position)
            );
            let (Some(a), Some(b)) = (
                a.payload.as_ref().and_then(CapturePayload::as_tensor),
                b.payload.as_ref().and_then(CapturePayload::as_tensor),
            ) else {
                panic!("measured additive evidence for action {action_index}: actual={a:?}, reference={b:?}")
            };
            assert_eq!(a.shape(), b.shape());
            let (
                eredu_core::TensorObservationData::F32(a),
                eredu_core::TensorObservationData::F32(b),
            ) = (a.data(), b.data())
            else {
                panic!("floating additive evidence")
            };
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                assert!(
                    (a - b).abs() <= 4e-4 + 4e-4 * b.abs(),
                    "summed intervention action {action_index}: {a} != {b}"
                );
            }
        };
        for ((token, actual), (expected_token, expected)) in actual.iter().zip(&expected) {
            assert_eq!(token, expected_token);
            assert_eq!(actual.records.len(), expected.records.len());
            assert_eq!(actual.interventions.len(), expected.interventions.len());
            for (a, b) in actual.records.iter().zip(&expected.records) {
                compare(a, b);
            }
            for (a, b) in actual.interventions.iter().zip(&expected.interventions) {
                assert_eq!((&a.operation_id, &a.outcome), (&b.operation_id, &b.outcome));
                assert_eq!(a.evidence.len(), b.evidence.len());
                if a.outcome == InterventionOutcome::Inactive {
                    for record in a.evidence.iter().chain(&b.evidence) {
                        assert_eq!(
                            record.outcome,
                            CaptureOutcome::Skipped {
                                reason: CaptureSkipReason::Schedule,
                            }
                        );
                        assert!(record.payload.is_none());
                    }
                    continue;
                }
                assert_eq!(a.outcome, InterventionOutcome::Applied);
                for (a, b) in a.evidence.iter().zip(&b.evidence) {
                    compare(a, b);
                }
            }
            assert!(actual
                .partitions
                .iter()
                .any(|evidence| evidence.combination == PartitionCaptureCombination::SumF64ToF32));
        }
    }
    runtime.synchronize().unwrap();
    reference.synchronize().unwrap();
}

// Capture only additive sources: completion must not rely on exporting logits.
// Nonlinear transforms must operate on the assembled write, not on each term.
fn verify_additive_component_transforms(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    reference: &mut ModelRuntime<MlxBackend<'_>>,
    capture: &eredu_core::capture::AdmittedCapturePlan,
    sampling: eredu_core::ResolvedGenerationConfig,
    targets: &[String],
) {
    use eredu_core::{capture::*, TextGenerationBackend as _};
    if targets.is_empty() {
        return;
    }
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    for transform in [
        CaptureTransform::Summary,
        CaptureTransform::Histogram {
            edges: vec![-1.713, -0.219, 0.037, 0.413, 1.913],
        },
    ] {
        let mut plan = capture.plan().clone();
        plan.selections.retain(|selection| {
            targets.iter().any(|target| {
                selection.path == *target || selection.path == format!("{target}.effective")
            })
        });
        assert!(!plan.selections.is_empty());
        for selection in &mut plan.selections {
            selection.transform = transform.clone();
            let axes = discovery
                .catalog
                .get(&selection.path)
                .unwrap()
                .axes
                .as_ref()
                .unwrap();
            let axis = axes.last().unwrap();
            let eredu_core::SymbolicDimension::Known(width) = axis.dimension else {
                panic!("known additive width")
            };
            selection.slices = vec![CaptureSlice {
                axis: axis.name.clone(),
                start: 1,
                end: width as u64,
                stride: 2,
            }];
        }
        let plan = plan
            .admit(
                &discovery.catalog,
                &discovery.support,
                &discovery.support.capture,
                capture.request(),
            )
            .unwrap();
        let run = |runtime: &mut ModelRuntime<MlxBackend<'_>>| {
            runtime.reset().unwrap();
            let mut generation = component_capture_generation(
                runtime,
                sampling,
                Some(eredu_core::TextPreparationOptions {
                    capture: Some(eredu_core::capture::SharedCapturePlan::new(plan.clone())),
                    interventions: None,
                }),
            )
            .unwrap();
            (0..3)
                .map(|_| {
                    let token = generation.next().unwrap().unwrap().token_id();
                    (token, generation.take_captured_delivery().unwrap().unwrap())
                })
                .collect::<Vec<_>>()
        };
        let expected = run(reference);
        let actual = run(runtime);
        for ((token, actual), (expected_token, expected)) in actual.iter().zip(&expected) {
            assert_eq!(token, expected_token);
            assert_eq!(actual.records.len(), plan.plan().selections.len());
            assert_eq!(actual.records.len(), expected.records.len());
            assert_eq!(actual.partitions.len(), actual.records.len());
            assert!(actual
                .partitions
                .iter()
                .all(|evidence| evidence.combination == PartitionCaptureCombination::SumF64ToF32));
            for (actual, expected) in actual.records.iter().zip(&expected.records) {
                assert_eq!(actual.path, expected.path);
                assert_eq!(actual.position, expected.position);
                assert_eq!(actual.selected_shape, expected.selected_shape);
                assert_eq!(actual.outcome, CaptureOutcome::Captured);
                assert_eq!(expected.outcome, CaptureOutcome::Captured);
                match (&actual.payload, &expected.payload) {
                    (Some(CapturePayload::Summary(a)), Some(CapturePayload::Summary(b))) => {
                        assert_eq!(
                            (
                                a.elements,
                                a.finite,
                                a.non_finite,
                                a.nan,
                                a.positive_infinity,
                                a.negative_infinity
                            ),
                            (
                                b.elements,
                                b.finite,
                                b.non_finite,
                                b.nan,
                                b.positive_infinity,
                                b.negative_infinity
                            )
                        );
                        for (a, b) in [a.min, a.max, a.mean, a.rms]
                            .into_iter()
                            .zip([b.min, b.max, b.mean, b.rms])
                        {
                            match (a, b) {
                                (Some(a), Some(b)) => {
                                    assert!((a - b).abs() <= 4e-4 + 4e-4 * b.abs())
                                }
                                (None, None) => (),
                                _ => panic!("matching finite summary"),
                            }
                        }
                    }
                    (Some(CapturePayload::Histogram(a)), Some(CapturePayload::Histogram(b))) => {
                        assert_eq!(a, b);
                    }
                    _ => panic!("matching summed transform: {actual:?}, {expected:?}"),
                }
            }
        }
    }
    runtime.synchronize().unwrap();
    reference.synchronize().unwrap();
}

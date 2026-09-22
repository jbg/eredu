fn verify_loaded_routed_interventions(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    reference: &mut ModelRuntime<MlxBackend<'_>>,
    capture: &eredu_core::capture::AdmittedCapturePlan,
    sampling: eredu_core::ResolvedGenerationConfig,
    baseline: &[(u32, eredu_core::capture::SharedCapturedStep)],
) {
    use eredu_core::{capture::*, intervention::*, TextGenerationBackend as _};
    let discovery = MlxBackend::intervention_discovery(runtime).unwrap();
    let points = discovery
        .points
        .iter()
        .filter(|point| point.routed_units.is_some())
        .collect::<Vec<_>>();
    assert!(!points.is_empty());
    for mode in 0..7 {
        eprintln!("public sparse intervention mode {mode}");
        let mut operations = Vec::new();
        for point in &points {
            let geometry = point.routed_units.as_ref().unwrap().geometry;
            let components = geometry.components().unwrap();
            for phase in [CapturePhase::Prefill, CapturePhase::Decode] {
                let position = u64::from(phase == CapturePhase::Prefill);
                let dtype = InterventionDtype::Float32;
                let indices = (0..geometry.experts)
                    .map(|expert| (expert * geometry.units_per_expert + 1) as u32)
                    .collect::<Vec<_>>();
                let payload = || InterventionTensor {
                    shape: vec![1, components],
                    values: InterventionValues::Float32(
                        (0..components)
                            .map(|column| 0.015 * (column % 7) as f32 - 0.04)
                            .collect(),
                    ),
                };
                let actions = match mode {
                    0 => vec![InterventionAction::MaskComponents {
                        dtype,
                        indices: (0..components as u32).collect(),
                        keep_selected: true,
                    }],
                    1 | 2 => vec![InterventionAction::MaskComponents {
                        dtype,
                        indices,
                        keep_selected: mode == 2,
                    }],
                    3 => vec![
                        InterventionAction::Scale { dtype, factor: 0.6 },
                        InterventionAction::Add { tensor: payload() },
                    ],
                    4 => vec![InterventionAction::Replace { tensor: payload() }],
                    5 => vec![InterventionAction::Zero { dtype }],
                    6 => vec![InterventionAction::Mask {
                        dtype,
                        shape: vec![1, components],
                        keep: (0..components).map(|column| column % 3 == 1).collect(),
                    }],
                    _ => unreachable!(),
                };
                for (ordinal, action) in actions.into_iter().enumerate() {
                    operations.push(InterventionOperation {
                        id: format!("{}-{phase:?}-{ordinal}", point.path),
                        target: point.path.clone(),
                        schedule: CaptureSchedule {
                            prefill: phase == CapturePhase::Prefill,
                            decode: phase == CapturePhase::Decode,
                            ..Default::default()
                        },
                        slices: vec![CaptureSlice {
                            axis: "token".into(),
                            start: position,
                            end: position + 1,
                            stride: 1,
                        }],
                        action,
                        evidence: InterventionEvidence::None,
                    });
                }
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
                    "native-sparse-trial",
                )
                .unwrap();
            MlxBackend::validate_text_interventions(runtime, capture, &admitted).unwrap();
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
        compare_sparse_trials(&actual, &expected);
        for ((_, actual), (_, expected)) in actual.iter().zip(&expected) {
            assert_eq!(actual.interventions.len(), expected.interventions.len());
            for (actual, expected) in actual.interventions.iter().zip(&expected.interventions) {
                assert_eq!(
                    actual.outcome, expected.outcome,
                    "sparse operation mode {mode}"
                );
                assert_eq!(
                    actual.routed_units, expected.routed_units,
                    "logical affected counts deduplicate peers and TP replicas"
                );
                assert!(actual.evidence.is_empty());
                if actual.outcome != InterventionOutcome::Inactive {
                    assert_eq!(
                        actual.outcome,
                        if mode == 0 {
                            InterventionOutcome::Unmatched
                        } else {
                            InterventionOutcome::Applied
                        }
                    );
                    let receipt = actual.routed_units.unwrap();
                    assert_eq!(receipt.completed_tokens, receipt.source_tokens);
                }
            }
        }
        if mode == 0 {
            compare_sparse_trials(&actual, baseline);
        } else {
            let logits = |step: &CapturedStep| {
                step.records
                    .iter()
                    .find(|r| r.path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
                    .unwrap()
                    .payload
                    .clone()
            };
            assert_ne!(
                logits(&actual[0].1),
                logits(&baseline[0].1),
                "mode {mode} changes the tested prediction"
            );
        }
        // A future child mask must activate sparse hooks even with no sparse
        // captures selected. Reuse the ordinary branch/re-admission driver.
        if mode == 2 {
            let mut host = capture.plan().clone();
            host.selections
                .retain(|selection| selection.path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH);
            let discovery = MlxBackend::capture_discovery(runtime).unwrap();
            let logits = host
                .admit(
                    &discovery.catalog,
                    &discovery.support,
                    &discovery.support.capture,
                    capture.request(),
                )
                .unwrap();
            verify_component_intervention_branches(runtime, &logits, &plan, sampling);
        }
        runtime.synchronize().unwrap();
        reference.synchronize().unwrap();
    }
}

fn compare_sparse_trials(
    actual: &[(u32, eredu_core::capture::SharedCapturedStep)],
    expected: &[(u32, eredu_core::capture::SharedCapturedStep)],
) {
    use eredu_core::capture::*;
    for ((token, actual), (expected_token, expected)) in actual.iter().zip(expected) {
        assert_eq!(token, expected_token);
        assert_eq!(actual.records.len(), expected.records.len());
        for (actual, expected) in actual.records.iter().zip(&expected.records) {
            assert_eq!(actual.path, expected.path);
            assert_eq!(actual.outcome, CaptureOutcome::Captured);
            assert_eq!(actual.selected_shape, expected.selected_shape);
            match (
                actual.payload.as_ref().unwrap(),
                expected.payload.as_ref().unwrap(),
            ) {
                (actual, expected)
                    if actual.as_tensor().is_some() && expected.as_tensor().is_some() =>
                {
                    routed_close(actual.as_tensor().unwrap(), expected.as_tensor().unwrap())
                }
                (CapturePayload::RoutedUnits(actual), CapturePayload::RoutedUnits(expected)) => {
                    assert_eq!(actual.geometry, expected.geometry);
                    assert_eq!(actual.rows.len(), expected.rows.len());
                    for (actual, expected) in actual.rows.iter().zip(&expected.rows) {
                        assert_eq!(
                            (
                                actual.token,
                                actual.slot,
                                actual.expert,
                                actual.unit_start,
                                actual.unit_stride
                            ),
                            (
                                expected.token,
                                expected.slot,
                                expected.expert,
                                expected.unit_start,
                                expected.unit_stride
                            )
                        );
                        assert!((actual.coefficient - expected.coefficient).abs() < 2e-5);
                        routed_close(&actual.values, &expected.values);
                    }
                }
                _ => panic!("matching sparse and logits records"),
            }
        }
    }
}

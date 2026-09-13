include!("routed/interventions.rs");

/// Public loaded discovery and controlled capture, compared with an independent
/// ordinary native session. No test observer or reconstructed expert values.
fn verify_loaded_routed_capture(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    reference: &mut ModelRuntime<MlxBackend<'_>>,
    template: &eredu_core::capture::AdmittedCapturePlan,
    sampling: eredu_core::ResolvedGenerationConfig,
) {
    use eredu_core::{
        capture::*, ObservationSupportStatus as S, ObservationValueType, TextGenerationBackend as _,
    };
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let points = discovery
        .catalog
        .points
        .iter()
        .filter(|point| matches!(point.value_type, ObservationValueType::RoutedUnits { .. }))
        .collect::<Vec<_>>();
    if points.is_empty() {
        return;
    }
    for point in &points {
        let support = discovery
            .support
            .points
            .iter()
            .find(|support| support.path == point.path)
            .unwrap();
        assert_eq!(
            support.prefill,
            S::Supported,
            "loaded sparse prefill {}",
            point.path
        );
        assert_eq!(
            support.decode,
            S::Supported,
            "loaded sparse decode {}",
            point.path
        );
    }
    assert!(discovery
        .support
        .capture
        .transformations
        .contains(&CaptureTransformKind::RoutedUnits));
    let interventions = MlxBackend::intervention_discovery(runtime).unwrap();
    for point in interventions
        .points
        .iter()
        .filter(|point| point.routed_units.is_some())
    {
        assert_eq!(point.prefill, S::Supported);
        assert_eq!(point.decode, S::Supported);
    }
    let mut host = template.plan().clone();
    host.selections = points
        .iter()
        .flat_map(|point| {
            let ObservationValueType::RoutedUnits { geometry, .. } = &point.value_type else {
                unreachable!()
            };
            [false, true].map(|sliced| CaptureSelection {
                id: format!("{}-{sliced}", point.path),
                path: point.path.clone(),
                schedule: CaptureSchedule::default(),
                slices: if sliced {
                    vec![CaptureSlice {
                        axis: "component".into(),
                        start: 1,
                        end: geometry.units_per_expert,
                        stride: 2,
                    }]
                } else {
                    vec![]
                },
                transform: CaptureTransform::RoutedUnits,
            })
        })
        .chain(
            template
                .plan()
                .selections
                .iter()
                .filter(|selection| selection.path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
                .cloned(),
        )
        .collect();
    let plan = host
        .admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            template.request(),
        )
        .unwrap();
    MlxBackend::validate_text_capture(runtime, &plan).unwrap();
    let run = |runtime: &mut ModelRuntime<MlxBackend<'_>>, plan: &AdmittedCapturePlan| {
        runtime.reset().unwrap();
        let mut generation = component_capture_generation(runtime, sampling);
        generation.enable_capture(plan.clone()).unwrap();
        (0..3)
            .map(|_| {
                let token = generation.next().unwrap().unwrap().token_id();
                (token, generation.take_captured_step().unwrap().unwrap())
            })
            .collect::<Vec<_>>()
    };
    let expected = run(reference, &plan);
    let actual = run(runtime, &plan);
    for ((token, step), (expected_token, expected)) in actual.iter().zip(&expected) {
        assert_eq!(token, expected_token);
        assert_eq!(step.records.len(), expected.records.len());
        assert_eq!(step.partitions.len(), step.records.len());
        for (selection_index, (record, expected)) in
            step.records.iter().zip(&expected.records).enumerate()
        {
            assert_eq!(record.path, expected.path);
            assert_eq!(record.outcome, CaptureOutcome::Captured);
            assert_eq!(record.selected_shape, expected.selected_shape);
            match (
                record.payload.as_ref().unwrap(),
                expected.payload.as_ref().unwrap(),
            ) {
                (CapturePayload::RoutedUnits(actual), CapturePayload::RoutedUnits(expected)) => {
                    assert_eq!(actual.geometry, expected.geometry);
                    assert!(
                        actual.source_token_ranges.is_empty(),
                        "global receipts retain receive chunks in producer evidence"
                    );
                    assert_eq!(actual.rows.len(), expected.rows.len());
                    assert!(!actual.rows.is_empty());
                    let evidence = step
                        .partitions
                        .iter()
                        .find(|e| e.context.selection_index == selection_index)
                        .unwrap();
                    let source = evidence
                        .contributions
                        .iter()
                        .filter_map(|producer| producer.routed.as_ref())
                        .map(|r| r.ownership.source_peer)
                        .collect::<std::collections::BTreeSet<_>>();
                    assert_eq!(source.len(), 1);
                    let source = source.into_iter().next().unwrap();
                    assert!(source.is_none() || source == Some(0));
                    for (row, expected) in actual.rows.iter().zip(&expected.rows) {
                        assert_eq!(row.source_peer, source);
                        assert_eq!(expected.source_peer, None);
                        assert_eq!(
                            (
                                row.token,
                                row.slot,
                                row.expert,
                                row.unit_start,
                                row.unit_stride
                            ),
                            (
                                expected.token,
                                expected.slot,
                                expected.expert,
                                expected.unit_start,
                                expected.unit_stride
                            )
                        );
                        assert!((row.coefficient - expected.coefficient).abs() < 2e-5);
                        routed_close(&row.values, &expected.values);
                    }
                }
                (CapturePayload::Tensor(actual), CapturePayload::Tensor(expected)) => {
                    routed_close(actual, expected)
                }
                _ => panic!("sparse units or final logits"),
            }
        }
    }
    runtime.synchronize().unwrap();
    reference.synchronize().unwrap();
    verify_component_capture_branches(runtime, &plan, sampling);
    verify_loaded_routed_interventions(runtime, reference, &plan, sampling, &actual);
    let mut skipped = plan.plan().clone();
    skipped.limits.on_limit = CaptureLimitPolicy::Skip;
    skipped.limits.per_step.captures = 0;
    let skipped = skipped
        .admit(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            plan.request(),
        )
        .unwrap();
    for ((token, step), (expected, _)) in run(runtime, &skipped).iter().zip(&actual) {
        assert_eq!(token, expected);
        assert!(step.partitions.is_empty());
        assert!(step.records.iter().all(|record| record.payload.is_none()
            && matches!(record.outcome, CaptureOutcome::Skipped { .. })));
    }
    runtime.synchronize().unwrap();
    eprintln!(
        "loaded sparse capture verified: {} selections, prefill and two decode steps, slices, replay and sibling isolation",
        plan.plan().selections.len()
    );
}

fn routed_close(actual: &eredu_core::TensorObservation, expected: &eredu_core::TensorObservation) {
    use eredu_core::TensorObservationData::F32;
    assert_eq!(actual.shape(), expected.shape());
    let (F32(actual), F32(expected)) = (actual.data(), expected.data()) else {
        panic!("F32 fixture")
    };

    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 2e-4 + 2e-4 * expected.abs(),
            "sparse/native {actual} != {expected}"
        );
    }
}

use super::*;

fn with_origin(source: &AdmittedCapturePlan, cached_positions: u64) -> AdmittedCapturePlan {
    let caps = CaptureCapabilities {
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
        capture: caps.clone(),
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
    source
        .plan()
        .clone()
        .admit_with_text_origin(
            &catalog,
            &support,
            &caps,
            source.request(),
            CaptureTextOrigin { cached_positions },
        )
        .unwrap()
}
#[test]
fn mismatched_origin_intervention_coupling_rejects_before_install_or_preflight() {
    let (original, intervention) = plans(
        vec![operation(
            "zero",
            InterventionAction::Zero {
                dtype: InterventionDtype::Float32,
            },
        )],
        true,
    );
    let capture = with_origin(&original, 7);
    let mut session = CaptureSession::new(capture.clone());
    assert!(matches!(
        preflight(&capture, &intervention, &Estimates),
        Err(CaptureError::Invalid(_))
    ));
    assert!(matches!(
        session.enable_interventions(intervention.clone(), std::sync::Arc::new(Estimates)),
        Err(CaptureError::Invalid(_))
    ));
    assert!(session.interventions.is_none());
    assert_eq!(session.cumulative_usage(), CaptureUsage::default());
    assert!(matches!(
        session.select_prediction_interventions(
            Some(intervention.clone()),
            std::sync::Arc::new(Estimates)
        ),
        Err(CaptureError::Invalid(_))
    ));
    assert!(session.interventions.is_none());
    assert_eq!(session.cumulative_usage(), CaptureUsage::default());
    let mut zero = CaptureSession::new(with_origin(&original, 0));
    zero.enable_interventions(intervention, std::sync::Arc::new(Estimates))
        .unwrap();
    assert!(zero.interventions.is_some());
    let (_, empty) = plans(vec![], true);
    session
        .enable_interventions(empty, std::sync::Arc::new(Estimates))
        .unwrap();
    assert!(session.interventions.is_none());
}

#[test]
fn matching_cached_origin_binds_original_host_and_canonical_prefill_positions() {
    let mut operation = operation(
        "scale",
        InterventionAction::Scale {
            dtype: InterventionDtype::Float32,
            factor: 2.0,
        },
    );
    operation.evidence = InterventionEvidence::None;
    let (original, old) = plans(vec![operation], true);
    let origin = CaptureTextOrigin {
        cached_positions: 7,
    };
    let capture = with_origin(&original, origin.cached_positions);
    let intervention = old
        .plan()
        .clone()
        .admit_with_text_origin(
            &session::discovery(&old),
            old.request(),
            origin,
            old.session_id(),
        )
        .unwrap();
    preflight(&capture, &intervention, &Estimates).unwrap();
    let mut run = CaptureSession::new(capture.clone());
    run.enable_interventions(intervention.clone(), std::sync::Arc::new(Estimates))
        .unwrap();
    assert!(run.interventions.is_some());
    let pool = crate::working_memory::WorkingMemoryPool::new(1 << 22, 7).unwrap();
    let source = pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&intervention).unwrap())
        .unwrap();
    let captured = SharedCapturePlan::new(capture);
    let host = crate::working_memory::CaptureRunHostPlan::prepare(&captured)
        .unwrap()
        .with_interventions(&source)
        .unwrap();
    assert!(host.intervention_source().unwrap().same_source(&source));
    let inference = InferenceGeometry {
        batch_size: 1,
        cached_positions: 7,
        input_positions: 2,
        prefill_chunk_positions: 1,
        max_output_tokens: 3,
        output: OutputDemand::Sequence,
    };
    for start in 0..2 {
        let chunk = crate::prefill::PrefillChunk {
            input: start..start + 1,
            position: 7 + start,
            output: inference.output.for_chunk(start + 1 == 2),
        };
        let window = InterventionPrefillWindow::new(&intervention, inference, &chunk).unwrap();
        window.validate(&intervention).unwrap();
        assert_eq!(window.range(), [start, start + 1]);
        assert_eq!(window.window().start, start);
        assert_eq!(window.window().logical_sequence, 2);
        assert_eq!(window.physical().sequence, 1);
        assert_eq!(window.physical().context, Some(8 + start));
        let mut wrong = chunk;
        wrong.position = start;
        assert!(InterventionPrefillWindow::new(&intervention, inference, &wrong).is_err());
    }
    let mut wrong = inference;
    wrong.cached_positions = 8;
    let chunk = crate::prefill::PrefillChunk {
        input: 0..1,
        position: 8,
        output: OutputDemand::Sequence,
    };
    assert!(InterventionPrefillWindow::new(&intervention, wrong, &chunk).is_err());
}

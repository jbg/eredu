use super::*;

#[test]
fn independent_model_capture_quotes_actual_context_scope_and_single_invocation() {
    let ordinary = admitted(
        vec![SymbolicDimension::Known(9), SymbolicDimension::Known(2)],
        CaptureTransform::FullTensor,
        vec![],
    );
    let mut point = ordinary.points()[0].clone();
    point.axes.as_mut().unwrap()[0].dimension = SymbolicDimension::Context;
    let catalog = ObservationCatalog {
        schema_version: 1,
        points: vec![point],
        completeness: DescriptionCompleteness::Complete,
    };
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: vec![ObservationSupport {
            path: "block.output".into(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }],
    };
    let capabilities = CaptureCapabilities {
        transformations: vec![CaptureTransformKind::FullTensor],
        max_histogram_bins: 0,
        physical_native_limit: false,
        conditions: vec![],
    };
    let mut plan = ordinary.plan().clone();
    plan.selections[0].schedule = CaptureSchedule::default();
    let source = SharedCapturePlan::new(
        plan.admit_invocations(
            &catalog,
            &support,
            &capabilities,
            CaptureInvocationBounds {
                batch: 1,
                max_sequence: 3,
                max_context: Some(9),
                max_predictions: 8,
            },
        )
        .unwrap(),
    );
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 6,
        input_positions: 3,
        max_output_tokens: 0,
        prefill_chunk_positions: 3,
        output: OutputDemand::Sequence,
    };
    let shape = CaptureInvocationShape {
        batch: 1,
        sequence: 3,
        context: Some(9),
    };
    let span = InferenceWorkspaceSpan::Prefill(eredu_runtime::prefill::PrefillChunk {
        input: 0..3,
        position: 6,
        output: OutputDemand::Sequence,
    });
    for selected in [[true], [false]] {
        let context = WorkspaceContext::new_recording_facts(Facts::default());
        let transfers = Cell::new(CaptureNativePopulation::default());
        let (mut observer, host) = CaptureWorkspaceObserver::with_invocation(
            &source,
            geometry,
            &context,
            &transfers,
            CapturePhase::Prefill,
            2,
            shape,
            &selected,
        )
        .unwrap();
        assert_eq!(host.claim_slots(), 2);
        assert_eq!(observer.requires_sequence_readout(), selected[0]);
        assert!(observer.begin_span(geometry, &span, 2, &context).unwrap());
        let (value, _backing) = imported(&context, &[9, 2], WorkspaceDtype::Float32, Some(72));
        context.begin_state_span([&value]).unwrap();
        observer.observe("block.output", &value).unwrap();
        assert_eq!(transfers.get().publications, 0);
        assert_eq!(transfers.get().completions, usize::from(selected[0]));
        assert_eq!(transfers.get().retained_roots, 5 * usize::from(selected[0]));
        if selected[0] {
            assert!(observer.roots.iter().any(|root| root.shape() == [9, 2]));
        } else {
            assert!(observer.roots.is_empty());
        }
        let mut retained = Vec::new();
        observer.visit_retained(&mut |root| retained.push(root.clone()));
        let report = context.finish_report(&retained).unwrap();
        assert_eq!(casts(&report), usize::from(selected[0]));
        observer.end_span(&span, &context).unwrap();
        assert!(observer.begin_span(geometry, &span, 2, &context).is_err());
    }
}

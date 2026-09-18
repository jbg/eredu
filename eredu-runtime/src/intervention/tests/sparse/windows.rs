use super::*;
use crate::{
    capture::{CaptureBackendProvider, SpeculativeCaptureObserver},
    inspection::{with_speculative_activation, SpeculativeActivationObserver},
};
use eredu_core::speculative::*;
struct Provider;
impl CaptureBackendProvider for Provider {
    type Tensor = Value;
    type Error = std::io::Error;
    type Backend<'a> = Backend;
    fn backend(&mut self) -> Backend {
        Backend::default()
    }
}
fn map(e: &CaptureExecutionError<std::io::Error>) -> String {
    e.to_string()
}
fn run() -> CaptureSession {
    let (capture, old) = plans_for(vec![InterventionAction::Scale {
        dtype: InterventionDtype::Float32,
        factor: 2.,
    }]);
    let bounds = CaptureInvocationBounds {
        batch: 1,
        max_sequence: 4,
        max_context: None,
        max_predictions: 1,
    };
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
    let capture = capture
        .plan()
        .clone()
        .admit_invocations(&catalog, &support, &support.capture, bounds)
        .unwrap();
    let mut discovery = session::discovery(&old);
    discovery.points[0].operations.push(InterventionKind::Add);
    let mut plan = old.plan().clone();
    plan.operations[0].slices = vec![CaptureSlice {
        axis: "token".into(),
        start: 1,
        end: 4,
        stride: 2,
    }];
    plan.operations[0].action = InterventionAction::Add {
        tensor: InterventionTensor {
            shape: vec![2, 6],
            values: InterventionValues::Float32((10..22).map(|v| v as f32).collect()),
        },
    };
    let mut run = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(capture));
    run.enable_interventions(
        plan.admit_invocations(&discovery, bounds, "session")
            .unwrap(),
        std::sync::Arc::new(Estimates),
    )
    .unwrap();
    run
}
fn batch<'a>(
    offset: usize,
    rows: usize,
    v: &'a Value,
    t: &'a Value,
    s: &'a Value,
    c: &'a Value,
    g: &'a Value,
) -> crate::RoutedUnitBatch<'a, Value> {
    crate::RoutedUnitBatch {
        units: eredu_nn::GroupedUnitBatch {
            values: v,
            group_indices: s,
            selection_indices: s,
            token_indices: t,
            coefficients: c,
            token_offset: offset,
            total_token_count: rows,
            group_count: 3,
        },
        source_groups: g,
        global_groups: None,
        provider_token_offset: 0,
        origins: None,
        unit_coordinates: None,
    }
}
#[test]
fn sparse_window_receipts_preserve_global_payloads_and_actual_native_chunk_order() {
    let v = Value {
        shape: vec![2, 2],
        data: vec![2., 3., 5., 7.],
    };
    let tokens = Value {
        shape: vec![2],
        data: vec![0., 0.],
    };
    let slots = Value {
        shape: vec![2],
        data: vec![1., 0.],
    };
    let coefficients = Value {
        shape: vec![1, 2],
        data: vec![0.3, 0.7],
    };
    let mut observer = SpeculativeCaptureObserver::new(
        run(),
        Provider,
        map as fn(&CaptureExecutionError<std::io::Error>) -> String,
        SpeculativeRequestId::new(0),
        vec![],
        vec![SpeculativeCaptureScope::Target],
    )
    .unwrap();
    observer.set_activation_origin(Some(SpeculativeActivationOrigin {
        request: SpeculativeRequestId::new(0),
        committed_tokens: 0,
        prediction: 0,
        prefix_digest: [9; 32],
        optimistic: false,
    }));
    observer.set_prefill_reduction_geometry(SpeculativePrefillReductionGeometry {
        target_sequence: 4,
        prediction_sequence: 4,
    });
    let mut values = Vec::new();
    for (a, b) in [(0, 2), (2, 4)] {
        observer.set_prefill_span(Some(SpeculativePrefillSpan {
            prompt_tokens: 4,
            input_start: a,
            input_end: b,
            position: a,
            hidden_start: a,
            token_start: a,
            sequence: b - a,
            seed_start: a,
        }));
        let groups = Value {
            shape: vec![2, 2],
            data: vec![1., 1., 0., 2.],
        };
        with_speculative_activation(
            Some(&mut observer),
            SpeculativeActivationPhase::TargetPrefill,
            2,
            |o| {
                let units = o.unwrap().routed_unit_observer("router")?.unwrap();
                for offset in 0..2 {
                    let out = units
                        .intervene(&batch(
                            offset,
                            2,
                            &v,
                            &tokens,
                            &slots,
                            &coefficients,
                            &groups,
                        ))
                        .map_err(|e| e.to_string())?;
                    values.extend(out.as_ref().unwrap_or(&v).data.clone());
                }
                Ok(())
            },
        )
        .unwrap();
    }
    observer.complete_prefill_reductions().unwrap();
    observer.finish_prefill_reductions(true);
    // Same actual ordinary source rows: duplicate expert slots and reversed
    // native slot order are preserved; logical rows1/3 choose distinct payloads.
    assert_eq!(
        values,
        [2., 3., 5., 7., 16., 18., 15., 18., 2., 3., 5., 7., 22., 24., 21., 24.]
    );
    let steps: Vec<_> = std::iter::from_fn(|| observer.take_activation_capture()).collect();
    let entry = &steps[1].prefill_reductions.as_ref().unwrap().interventions[0];
    assert_eq!(entry.status, SpeculativePrefillReductionStatus::Complete);
    assert_eq!(
        entry.record.routed_units,
        Some(RoutedUnitInterventionReceipt {
            source_tokens: 4,
            completed_tokens: 4,
            affected_values: 8
        })
    );
    assert!(entry.record.evidence.is_empty());
    for s in &steps {
        assert_eq!(
            s.captures.as_step().interventions[0]
                .routed_units
                .unwrap()
                .completed_tokens,
            2
        );
    }
}
#[test]
fn sparse_window_duplicate_or_missing_native_chunk_cannot_publish_completion() {
    for duplicate in [false, true] {
        let mut run = run();
        let mut backend = Backend::default();
        run.begin_invocation_window(
            CapturePhase::Prefill,
            0,
            CaptureInvocationShape {
                batch: 1,
                sequence: 2,
                context: None,
            },
            crate::capture::CaptureInvocationSelection::default(),
            CaptureInvocationWindow {
                logical_sequence: 4,
                start: 2,
            },
        )
        .unwrap();
        chunk(&mut run, &mut backend, 0, vec![1., 1., 0., 2.]).unwrap();
        let usage = run.cumulative_usage();
        if duplicate {
            assert!(chunk(&mut run, &mut backend, 0, vec![1., 1., 0., 2.]).is_err());
        }
        assert!(run.finish_interventions().is_err());
        assert_eq!(run.cumulative_usage(), usage);
        assert_ne!(
            run.take_step().unwrap().interventions[0].outcome,
            InterventionOutcome::Applied
        );
    }
}

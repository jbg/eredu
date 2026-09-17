use super::*;
use eredu_checkpoint::{BlockFp8Format, BlockFp8ScaleEncoding, LinearFormat};
use eredu_nn::{
    LinearOperator, NeuralBackend, ParameterSpec, ProjectionInputObserver,
    RetainedGeneratedTensorFactory,
};

#[derive(Debug)]
struct GeneratedFacts {
    missing: bool,
}
impl WorkspaceMechanisms for GeneratedFacts {
    fn projection_input_observation_mechanism(
        &self,
        _: &eredu_nn::LinearFormatSpec,
    ) -> Result<Option<eredu_nn::ProjectionInputObservationMechanism>> {
        Ok(Some(
            eredu_nn::ProjectionInputObservationMechanism::BlockFp8Gpu,
        ))
    }
    fn operation_bound(&self, op: &WorkspaceOperation) -> Result<Option<WorkspaceOperationBound>> {
        if self.missing && matches!(op.kind, WorkspaceOperationKind::BlockFp8ActivationDecode) {
            return Ok(None);
        }
        Ok(Some(WorkspaceOperationBound {
            outputs: op
                .outputs
                .iter()
                .map(|o| o.bytes().map(WorkspaceOutputStorage::Allocate))
                .collect::<Result<Vec<_>>>()?,
            scratch_bytes: 0,
            assumptions: "test owned outputs for actual generated operator shapes".into(),
        }))
    }
    fn host_workspace_bound(&self, _: &WorkspaceOperation) -> Result<Option<WorkspaceHostBound>> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "test no host numerical workspace".into(),
        }))
    }
}
struct Projection<'a, 'p> {
    capture: &'a mut CaptureWorkspaceObserver<'p>,
    path: &'static str,
}
impl ProjectionInputObserver<WorkspaceTensor> for Projection<'_, '_> {
    fn observe(&mut self, _: &WorkspaceTensor) -> Result<()> {
        panic!("actual FP8 generated producer")
    }
    fn observe_generated(
        &mut self,
        _: &WorkspaceTensor,
        _: &eredu_nn::GeneratedTensorSource,
        _: &mut dyn FnMut() -> Result<WorkspaceTensor>,
    ) -> Result<()> {
        panic!("retained protocol required")
    }
    fn observe_generated_retained(
        &mut self,
        p: &WorkspaceTensor,
        s: &eredu_nn::GeneratedTensorSource,
        f: &mut dyn RetainedGeneratedTensorFactory<WorkspaceTensor, eredu_nn::Error>,
    ) -> Result<()> {
        self.capture.observe_generated_retained(
            self.path,
            p,
            &eredu_runtime::capture::generated_capture_source(s),
            f,
        )
    }
}
fn linear(context: &WorkspaceContext, width: i32) -> WorkspaceLinear {
    WorkspaceBackend::linear(
        eredu_nn::LinearSpec {
            input: width,
            output: 3,
            weight: ParameterSpec::trainable("matrix.weight").unwrap(),
            bias: None,
            format: eredu_nn::LinearFormatSpec::scaled(
                LinearFormat::E4M3BlockFp8(
                    BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint).unwrap(),
                ),
                ParameterSpec::trainable("matrix.scales").unwrap(),
            )
            .unwrap(),
        },
        context,
    )
    .unwrap()
}
fn two_hooks(width: usize, preview: usize) -> SharedCapturePlan {
    let a = admitted(
        vec![SymbolicDimension::Known(2), SymbolicDimension::Known(width)],
        CaptureTransform::Preview {
            max_elements: preview as u64,
        },
        vec![],
    );
    let mut raw = a.plan().clone();
    let mut points = a.points().to_vec();
    points[0].path = "first.input".into();
    raw.selections[0].path = points[0].path.clone();
    let mut second = points[0].clone();
    second.path = "second.input".into();
    points.push(second);
    let mut selected = raw.selections[0].clone();
    selected.path = "second.input".into();
    selected.id = "second".into();
    raw.selections.push(selected);
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: points
            .iter()
            .map(|p| ObservationSupport {
                path: p.path.clone(),
                prefill: ObservationSupportStatus::Supported,
                decode: ObservationSupportStatus::Supported,
                floating_to_f32: true,
            })
            .collect(),
    };
    SharedCapturePlan::new(
        raw.admit(
            &ObservationCatalog {
                schema_version: 1,
                points,
                completeness: DescriptionCompleteness::Complete,
            },
            &support,
            &CaptureCapabilities {
                transformations: vec![CaptureTransformKind::Preview],
                max_histogram_bins: 0,
                physical_native_limit: false,
                conditions: vec![],
            },
            CaptureRequestShape {
                batch: 1,
                prompt_tokens: 3,
                max_predictions: 4,
            },
        )
        .unwrap(),
    )
}
fn decodes(r: &WorkspaceTraceReport) -> usize {
    r.operations
        .iter()
        .filter(|o| matches!(o.kind, WorkspaceOperationKind::BlockFp8ActivationDecode))
        .count()
}
#[test]
fn actual_generated_hooks_keep_all_roots_through_later_finish_and_clear_only_at_span_end() {
    for width in [1, 128, 259] {
        let context = WorkspaceContext::new(GeneratedFacts { missing: false });
        let source = two_hooks(width, 2);
        let (mut observer, _) =
            CaptureWorkspaceObserver::new(&source, geometry(), &context).unwrap();
        begin(&mut observer, &context);
        let input = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[2, width as i32], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        let mut a = linear(&context, width as i32);
        let mut b = linear(&context, width as i32);
        context.begin_state_span([&input]).unwrap();
        // begin() already entered the first decode span and checked its metadata.
        let first = a
            .forward_with_input_observer(
                &input,
                &context,
                Some(&mut Projection {
                    capture: &mut observer,
                    path: "first.input",
                }),
            )
            .unwrap();
        assert_eq!(observer.roots.len(), 10);
        let old = observer.roots.clone();
        let second = b
            .forward_with_input_observer(
                &input,
                &context,
                Some(&mut Projection {
                    capture: &mut observer,
                    path: "second.input",
                }),
            )
            .unwrap();
        assert_eq!(observer.roots.len(), 20);
        for (a, b) in old.iter().zip(&observer.roots) {
            assert_eq!(
                context
                    .report(&[a.clone(), b.clone()])
                    .unwrap()
                    .retained_bytes,
                context
                    .report(std::slice::from_ref(a))
                    .unwrap()
                    .retained_bytes
            );
        }
        let mut roots = vec![first, second];
        observer.visit_retained(&mut |v| roots.push(v.clone()));
        let report = context.report(&roots).unwrap();
        assert_eq!(decodes(&report), 2);
        assert!(report.total_bytes.is_some());
        assert_eq!(
            report
                .operations
                .iter()
                .filter(|op| matches!(op.kind, WorkspaceOperationKind::ProjectionFinish(_)))
                .count(),
            2
        );
        assert!(
            report.total_bytes.unwrap()
                >= roots
                    .iter()
                    .map(|v| v.layout().bytes().unwrap())
                    .sum::<u64>()
        );
        observer.end_span(&decode(1), &context).unwrap();
        assert!(observer.roots.is_empty());
    }
}
#[test]
fn generated_unknown_facts_stay_unknown_and_preview_zero_still_constructs() {
    for missing in [false, true] {
        let context = WorkspaceContext::new(GeneratedFacts { missing });
        let source = two_hooks(259, 0);
        let (mut observer, _) =
            CaptureWorkspaceObserver::new(&source, geometry(), &context).unwrap();
        begin(&mut observer, &context);
        let input = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[2, 259], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        let mut layer = linear(&context, 259);
        context.begin_state_span([&input]).unwrap();
        // begin() already entered the first decode span and checked its metadata.
        let output = layer
            .forward_with_input_observer(
                &input,
                &context,
                Some(&mut Projection {
                    capture: &mut observer,
                    path: "first.input",
                }),
            )
            .unwrap();
        let r = context.report(&[output]).unwrap();
        assert_eq!(decodes(&r), 1);
        assert_eq!(r.total_bytes.is_none(), missing);
        assert_eq!(observer.roots.len(), 10);
        assert!(observer.ledger.total().retained_bytes > 0);
        observer.end_span(&decode(1), &context).unwrap();
        assert!(observer.roots.is_empty());
    }
}

#[test]
fn independent_generated_capture_counts_one_factory_and_every_selected_transfer() {
    let ordinary = two_hooks(259, 7);
    let mut raw = ordinary.admission().plan().clone();
    raw.selections[1].path = raw.selections[0].path.clone();
    let support = ObservationSupportReport {
        schema_version: 1,
        capture: Default::default(),
        points: ordinary.admission().points().iter().map(|point| ObservationSupport {
            path: point.path.clone(),
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            floating_to_f32: true,
        }).collect(),
    };
    let source = SharedCapturePlan::new(raw.admit_invocations(
        &ObservationCatalog {
            schema_version: 1,
            points: ordinary.admission().points().to_vec(),
            completeness: DescriptionCompleteness::Complete,
        },
        &support,
        &CaptureCapabilities {
            transformations: vec![CaptureTransformKind::Preview],
            max_histogram_bins: 0,
            physical_native_limit: false,
            conditions: vec![],
        },
        CaptureInvocationBounds { batch: 1, max_sequence: 3, max_context: None, max_predictions: 8 },
    ).unwrap());
    for selected in [[true, true], [true, false], [false, false]] {
        let context = WorkspaceContext::new(GeneratedFacts { missing: false });
        let transfers = Cell::new(CaptureNativePopulation::default());
        let shape = CaptureInvocationShape { batch: 1, sequence: 1, context: None };
        let geometry = InferenceGeometry {
            batch_size: 1, cached_positions: 7, input_positions: 1,
            max_output_tokens: 0, prefill_chunk_positions: 1,
            output: OutputDemand::LastPosition,
        };
        // Every embedded equation is a single physical input span, independently
        // of its logical Decode capture phase and scheduler coordinate.
        let span = InferenceWorkspaceSpan::Prefill(eredu_runtime::prefill::PrefillChunk {
            input: 0..1, position: 7, output: OutputDemand::LastPosition,
        });
        let (mut observer, _) = CaptureWorkspaceObserver::with_invocation(
            &source, geometry, &context, &transfers,
            CapturePhase::Decode, 2, shape, &selected,
        ).unwrap();
        assert!(observer.begin_span(geometry, &span, 2, &context).unwrap());
        let input = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[2, 259], WorkspaceDtype::Float32).unwrap(), &context,
        ).unwrap();
        context.begin_state_span([&input]).unwrap();
        let mut layer = linear(&context, 259);
        let output = layer.forward_with_input_observer(
            &input, &context,
            Some(&mut Projection { capture: &mut observer, path: "first.input" }),
        ).unwrap();
        let count = selected.into_iter().filter(|selected| *selected).count();
        let population = transfers.get();
        assert_eq!(population.publications, 0);
        assert_eq!(population.completions, count);
        assert_eq!(population.retained_roots, if count == 0 { 0 } else { 9 + 5 * count });
        let mut roots = vec![output];
        observer.visit_retained(&mut |value| roots.push(value.clone()));
        let report = context.report(&roots).unwrap();
        assert_eq!(decodes(&report), usize::from(count != 0));
        assert!(report.total_bytes.is_some());
        if count != 0 {
            // Both rows use the exact same generated value; compact operands and
            // all seven intermediates survive until the enclosing phase ends.
            assert!(observer.roots.iter().any(|root| root.shape() == [2, 259] && root.layout().dtype() == WorkspaceDtype::Uint8));
            assert!(observer.roots.iter().any(|root| root.shape() == [2, 3]));
            assert!(population.controls > CaptureNativePopulation::within_raw().unwrap().controls * count);
        }
        observer.end_span(&span, &context).unwrap();
        assert!(observer.roots.is_empty());
    }
}

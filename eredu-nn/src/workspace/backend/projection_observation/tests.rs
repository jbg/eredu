use super::*;
use eredu_checkpoint::{BlockFp8Format, BlockFp8ScaleEncoding};

#[derive(Debug)]
struct Facts {
    selected: bool,
    decode: bool,
    host: bool,
}
impl WorkspaceMechanisms for Facts {
    fn projection_input_observation_mechanism(
        &self,
        format: &crate::LinearFormatSpec,
    ) -> Result<Option<ProjectionInputObservationMechanism>, Error> {
        Ok(
            if matches!(format.encoding(), LinearFormat::E4M3BlockFp8(_)) {
                self.selected
                    .then_some(ProjectionInputObservationMechanism::BlockFp8Gpu)
            } else {
                Some(ProjectionInputObservationMechanism::Borrowed)
            },
        )
    }
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        if !self.decode && matches!(op.kind, WorkspaceOperationKind::BlockFp8ActivationDecode) {
            return Ok(None);
        }
        Ok(Some(WorkspaceOperationBound {
            outputs: op
                .outputs
                .iter()
                .map(|x| x.bytes().map(WorkspaceOutputStorage::Allocate))
                .collect::<Result<_, _>>()?,
            scratch_bytes: 3,
            assumptions: "test fixed allocation and three scratch bytes per operation".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(self.host.then(|| WorkspaceHostBound {
            bytes: 2,
            assumptions: "test two-byte sidecar".into(),
        }))
    }
}
fn linear(context: &WorkspaceContext, fp8: bool, width: i32) -> WorkspaceLinear {
    let format = if fp8 {
        crate::LinearFormatSpec::scaled(
            LinearFormat::E4M3BlockFp8(
                BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint).unwrap(),
            ),
            ParameterSpec::trainable("matrix.scales").unwrap(),
        )
        .unwrap()
    } else {
        crate::LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap()
    };
    WorkspaceBackend::linear(
        LinearSpec {
            input: width,
            output: 3,
            weight: ParameterSpec::trainable("matrix.weight").unwrap(),
            bias: None,
            format,
        },
        context,
    )
    .unwrap()
}
struct Observer {
    selected: bool,
    borrowed: usize,
    offered: usize,
    value: Option<WorkspaceTensor>,
    source: Option<crate::GeneratedTensorSource>,
}
impl Observer {
    fn new(selected: bool) -> Self {
        Self {
            selected,
            borrowed: 0,
            offered: 0,
            value: None,
            source: None,
        }
    }
}
impl crate::ProjectionInputObserver<WorkspaceTensor> for Observer {
    fn observe(&mut self, input: &WorkspaceTensor) -> Result<(), Error> {
        self.borrowed += 1;
        self.value = Some(input.clone());
        Ok(())
    }
    fn observe_generated(
        &mut self,
        input: &WorkspaceTensor,
        source: &crate::GeneratedTensorSource,
        factory: &mut dyn FnMut() -> Result<WorkspaceTensor, Error>,
    ) -> Result<(), Error> {
        self.offered += 1;
        self.source = Some(*source);
        if self.selected {
            let value = factory()?;
            assert!(value.same_context(input));
            assert_eq!(value.shape(), input.shape());
            self.value = Some(value)
        }
        Ok(())
    }
}
#[test]
fn dense_borrows_and_fp8_selected_factory_traces_fixed_program_without_skipped_creation() {
    for shape in [&[2, 259][..], &[129, 1][..], &[1, 2, 256][..]] {
        for fp8 in [false, true] {
            for selected in [false, true] {
                let context = WorkspaceContext::new(Facts {
                    selected: true,
                    decode: true,
                    host: true,
                });
                let mut linear = linear(&context, fp8, *shape.last().unwrap());
                let input = WorkspaceTensor::existing(
                    WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap(),
                    &context,
                )
                .unwrap();
                context.begin_state_span([&input]).unwrap();
                let mut observer = Observer::new(selected);
                let output = linear
                    .forward_with_input_observer(&input, &context, Some(&mut observer))
                    .unwrap();
                let report = context.report(&[output]).unwrap();
                assert_eq!(observer.borrowed, usize::from(!fp8));
                assert_eq!(observer.offered, usize::from(fp8));
                if !fp8 {
                    assert_eq!(report.operations.len(), 1);
                    assert!(matches!(
                        report.operations[0].kind,
                        WorkspaceOperationKind::Projection(_)
                    ));
                } else {
                    assert!(matches!(
                        report.operations.first().unwrap().kind,
                        WorkspaceOperationKind::ProjectionPrepare(_)
                    ));
                    assert!(matches!(
                        report.operations.last().unwrap().kind,
                        WorkspaceOperationKind::ProjectionFinish(_)
                    ));
                    assert_eq!(report.operations.len(), if selected { 9 } else { 2 });
                    assert_eq!(
                        observer.source.unwrap(),
                        BlockFp8InputReconstructionPlan::new(shape)
                            .unwrap()
                            .logical_capture_source()
                            .unwrap()
                    );
                    if selected {
                        assert!(matches!(
                            report.operations[1].kind,
                            WorkspaceOperationKind::View("expand_dims")
                        ));
                        assert!(matches!(
                            report.operations[2].kind,
                            WorkspaceOperationKind::View("broadcast")
                        ));
                        assert!(matches!(
                            report.operations[3].kind,
                            WorkspaceOperationKind::View("reshape")
                        ));
                        assert!(matches!(
                            report.operations[4].kind,
                            WorkspaceOperationKind::BlockFp8ActivationDecode
                        ));
                        assert!(matches!(
                            report.operations[5].kind,
                            WorkspaceOperationKind::StaticSlice { .. }
                        ));
                        assert!(matches!(
                            report.operations[6].kind,
                            WorkspaceOperationKind::Elementwise("multiply")
                        ));
                        assert!(matches!(
                            report.operations[7].kind,
                            WorkspaceOperationKind::View("reshape")
                        ));
                        assert_eq!(
                            observer.value.as_ref().unwrap().layout().dtype(),
                            WorkspaceDtype::Float32
                        );
                    }
                }
                assert_eq!(input.shape(), shape);
                assert!(report.total_bytes.is_some());
            }
        }
    }
}
#[test]
fn unknown_selected_mechanism_rejects_before_trace_and_missing_operation_or_host_stays_unknown() {
    for (selected, decode, host) in [
        (false, true, true),
        (true, false, true),
        (true, true, false),
    ] {
        let context = WorkspaceContext::new(Facts {
            selected,
            decode,
            host,
        });
        let mut linear = linear(&context, true, 259);
        let input = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[2, 259], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        context.begin_state_span([&input]).unwrap();
        let mut observer = Observer::new(true);
        let output = linear.forward_with_input_observer(&input, &context, Some(&mut observer));
        if !selected {
            assert!(output.is_err());
            assert_eq!(observer.offered, 0);
            assert!(context.report(&[]).unwrap().operations.is_empty());
        } else {
            let report = context
                .report(&[output.unwrap(), observer.value.take().unwrap()])
                .unwrap();
            assert!(report.total_bytes.is_none());
        }
    }
}

mod retained;

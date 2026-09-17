use super::*;
use eredu_checkpoint::{BlockFp8Format, BlockFp8ScaleEncoding, LinearFormat};
use eredu_nn::{LinearOperator, LinearSpec, NeuralBackend, ParameterSpec, Tensor};

fn selected() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: MetalAllocationFacts { page_size: 16384 },
        sdpa_blocks: None,
    }
}
fn format(encoding: BlockFp8ScaleEncoding) -> LinearFormatSpec {
    LinearFormatSpec::scaled(
        LinearFormat::E4M3BlockFp8(BlockFp8Format::new(128, 128, encoding).unwrap()),
        ParameterSpec::trainable("matrix.scales").unwrap(),
    )
    .unwrap()
}
fn full(width: i32, rows: i32, encoding: BlockFp8ScaleEncoding, bias: bool) -> WorkspaceOperation {
    let format = format(encoding);
    let mut inputs = vec![
        WorkspaceLayout::new(&[rows, width], WorkspaceDtype::Float32).unwrap(),
        WorkspaceLayout::new(&[129, width], WorkspaceDtype::Uint8).unwrap(),
        WorkspaceLayout::new(
            &[2, (width as u64).div_ceil(128) as i32],
            if encoding == BlockFp8ScaleEncoding::Ue8m0 {
                WorkspaceDtype::Uint8
            } else {
                WorkspaceDtype::Float32
            },
        )
        .unwrap(),
    ];
    if bias {
        inputs.push(WorkspaceLayout::new(&[129], WorkspaceDtype::Float32).unwrap())
    }
    WorkspaceOperation {
        kind: WorkspaceOperationKind::Projection(format),
        inputs,
        outputs: vec![WorkspaceLayout::new(&[rows, 129], WorkspaceDtype::Float32).unwrap()],
    }
}
fn bound_total(bound: &WorkspaceOperationBound) -> u64 {
    bound.scratch_bytes
        + bound
            .outputs
            .iter()
            .map(|effect| match effect {
                WorkspaceOutputStorage::Allocate(n) => *n,
                other => panic!("unexpected {other:?}"),
            })
            .sum::<u64>()
}
fn phases(whole: &WorkspaceOperation) -> (WorkspaceOperation, WorkspaceOperation) {
    let WorkspaceOperationKind::Projection(format) = &whole.kind else {
        unreachable!()
    };
    let plan = eredu_nn::BlockFp8InputReconstructionPlan::new(whole.inputs[0].shape()).unwrap();
    let roots = vec![
        WorkspaceLayout::new(&plan.values_shape(), WorkspaceDtype::Uint8).unwrap(),
        WorkspaceLayout::new(&plan.scales_shape(), WorkspaceDtype::Float32).unwrap(),
    ];
    let prepare = WorkspaceOperation {
        kind: WorkspaceOperationKind::ProjectionPrepare(format.clone()),
        inputs: whole.inputs.clone(),
        outputs: roots.clone(),
    };
    let mut inputs = whole.inputs.clone();
    inputs.extend(roots);
    let finish = WorkspaceOperation {
        kind: WorkspaceOperationKind::ProjectionFinish(format.clone()),
        inputs,
        outputs: whole.outputs.clone(),
    };
    (prepare, finish)
}
#[test]
fn selected_phase_sum_equals_existing_whole_envelope_with_exact_scale_host_lifetime() {
    let facts = selected();
    for (rows, width) in [(2, 259), (129, 1), (3, 256)] {
        for encoding in [
            BlockFp8ScaleEncoding::FloatingPoint,
            BlockFp8ScaleEncoding::Ue8m0,
        ] {
            for bias in [false, true] {
                let whole = full(width, rows, encoding, bias);
                let (prepare, finish) = phases(&whole);
                assert_eq!(
                    bound_total(&facts.operation_bound(&whole).unwrap().unwrap()),
                    bound_total(&facts.operation_bound(&prepare).unwrap().unwrap())
                        + bound_total(&facts.operation_bound(&finish).unwrap().unwrap())
                );
                let host = facts.host_workspace_bound(&whole).unwrap().unwrap().bytes;
                assert_eq!(
                    host,
                    facts.host_workspace_bound(&prepare).unwrap().unwrap().bytes
                );
                assert_eq!(
                    facts.host_workspace_bound(&finish).unwrap().unwrap().bytes,
                    0
                );
                assert_eq!(host, 0);
            }
        }
    }
}
struct Capture {
    context: WorkspaceContext,
    value: Option<WorkspaceTensor>,
    before_finish: Option<u64>,
}
impl eredu_nn::ProjectionInputObserver<WorkspaceTensor> for Capture {
    fn observe(&mut self, _: &WorkspaceTensor) -> Result<(), Error> {
        panic!("FP8 must offer generated input")
    }
    fn observe_generated(
        &mut self,
        prototype: &WorkspaceTensor,
        source: &eredu_nn::GeneratedTensorSource,
        create: &mut dyn FnMut() -> Result<WorkspaceTensor, Error>,
    ) -> Result<(), Error> {
        let value = create()?;
        assert_eq!(value.shape(), prototype.shape());
        assert_eq!(source.element_type, Some(eredu_nn::TensorElementType::F32));
        self.before_finish = self.context.report(&[value.clone()])?.total_bytes;
        self.value = Some(value);
        Ok(())
    }
}
#[test]
fn actual_selected_reconstruction_and_prepare_roots_coexist_with_finish_output() {
    let facts = selected();
    let context = WorkspaceContext::new(facts);
    let mut linear = WorkspaceBackend::linear(
        LinearSpec {
            input: 259,
            output: 129,
            weight: ParameterSpec::trainable("matrix.weight").unwrap(),
            bias: Some(ParameterSpec::trainable("matrix.bias").unwrap()),
            format: format(BlockFp8ScaleEncoding::Ue8m0),
        },
        &context,
    )
    .unwrap();
    let input = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[2, 259], WorkspaceDtype::Float32).unwrap(),
        &context,
    )
    .unwrap();
    context.begin_state_span([&input]).unwrap();
    let mut observer = Capture {
        context: context.clone(),
        value: None,
        before_finish: None,
    };
    let output = linear
        .forward_with_input_observer(&input, &context, Some(&mut observer))
        .unwrap();
    let captured = observer.value.take().unwrap();
    let report = context.report(&[output.clone(), captured.clone()]).unwrap();
    let finish = report.operations.last().unwrap();
    assert!(matches!(
        finish.kind,
        WorkspaceOperationKind::ProjectionFinish(_)
    ));
    assert_eq!(
        report.total_bytes.unwrap(),
        observer.before_finish.unwrap()
            + bound_total(&facts.operation_bound(finish).unwrap().unwrap())
    );
    assert_eq!(report.host_workspace_bytes, Some(0));
    let only_output = context.report(&[output]).unwrap().retained_bytes.unwrap();
    assert!(
        report.retained_bytes.unwrap() > only_output,
        "generated multiplication payload stays live with final projection output"
    );
    assert_eq!(
        report
            .operations
            .iter()
            .filter(|x| matches!(x.kind, WorkspaceOperationKind::ProjectionPrepare(_)))
            .count(),
        1
    );
    let decode = report
        .operations
        .iter()
        .find(|x| matches!(x.kind, WorkspaceOperationKind::BlockFp8ActivationDecode))
        .unwrap();
    assert_eq!(decode.inputs[0].dtype(), WorkspaceDtype::Uint8);
    assert_eq!(decode.outputs[0].dtype(), WorkspaceDtype::Float32);
    assert_eq!(captured.shape(), [2, 259]);
}
#[test]
fn malformed_prepared_roles_and_decode_descriptors_reject_tensor_and_host_facts() {
    let facts = selected();
    let whole = full(259, 2, BlockFp8ScaleEncoding::FloatingPoint, false);
    let (prepare, finish) = phases(&whole);
    let mut invalid = Vec::new();
    let mut op = prepare.clone();
    op.outputs.pop();
    invalid.push(op);
    let mut op = finish.clone();
    op.inputs.pop();
    invalid.push(op);
    let mut op = finish;
    let last = op.inputs.len() - 1;
    op.inputs[last] = WorkspaceLayout::new(&[2, 3], WorkspaceDtype::Uint8).unwrap();
    invalid.push(op);
    for dtype in [WorkspaceDtype::Int32, WorkspaceDtype::Float32] {
        invalid.push(WorkspaceOperation {
            kind: WorkspaceOperationKind::BlockFp8ActivationDecode,
            inputs: vec![WorkspaceLayout::new(&[2, 259], dtype).unwrap()],
            outputs: vec![WorkspaceLayout::new(&[2, 259], WorkspaceDtype::Float32).unwrap()],
        })
    }
    for op in invalid {
        assert!(facts.operation_bound(&op).is_err());
        assert!(facts.host_workspace_bound(&op).is_err());
    }
}

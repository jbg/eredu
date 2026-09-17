use super::*;
use eredu_checkpoint::AffineQuantization;
use eredu_nn::ParameterSpec;

fn floating_layout(shape: &[i32], dtype: Option<F>) -> WorkspaceLayout {
    WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap()
        .with_representation(dtype.map(|value| WorkspaceRepresentation::new(value, true)))
}
fn operation(input: Option<F>, scale: Option<F>, affine_bias: Option<F>) -> WorkspaceOperation {
    WorkspaceOperation {
        kind: WorkspaceOperationKind::Projection(LinearFormatSpec::affine(
            LinearFormat::Affine(AffineQuantization::new(64, 4).unwrap()),
            ParameterSpec::trainable("projection.scales").unwrap(),
            ParameterSpec::trainable("projection.biases").unwrap(),
        ).unwrap()),
        inputs: vec![
            floating_layout(&[1, 2, 64], input),
            WorkspaceLayout::new(&[32, 8], WorkspaceDtype::Uint32).unwrap(),
            floating_layout(&[32, 1], scale),
            floating_layout(&[32, 1], affine_bias),
        ],
        outputs: vec![floating_layout(&[1, 2, 32], None)],
    }
}
fn scalar(operation: &WorkspaceOperation) -> Option<F> {
    super::super::output(operation.as_view(), 0).map(WorkspaceRepresentation::dtype)
}

#[test]
fn affine_projection_requires_actual_companion_sources_and_keeps_replacement_branch() {
    for (input, scale, bias, expected) in [
        (F::Float16, F::Float16, F::Float16, F::Float16),
        (F::Bfloat16, F::Bfloat16, F::Bfloat16, F::Bfloat16),
        (F::Float32, F::Float32, F::Float32, F::Float32),
        (F::Float16, F::Float32, F::Float16, F::Float32),
        (F::Float16, F::Float16, F::Bfloat16, F::Float32),
    ] {
        let op = operation(Some(input), Some(scale), Some(bias));
        assert_eq!(scalar(&op), Some(expected));
        assert!(!super::super::output(op.as_view(), 0).unwrap().row_contiguous());
        for ordinal in [0, 2, 3] {
            let mut unknown = op.clone();
            unknown.inputs[ordinal] = unknown.inputs[ordinal].clone().with_representation(None);
            assert_eq!(scalar(&unknown), None, "missing actual floating operand {ordinal}");
        }
    }
    let mut op = operation(Some(F::Float16), Some(F::Float32), Some(F::Float32));
    op.inputs[1] = floating_layout(&[32, 64], Some(F::Float16));
    assert_eq!(scalar(&op), Some(F::Float16), "unused packed companions cannot widen replacement matmul");
    op.inputs.push(floating_layout(&[32], Some(F::Float32)));
    assert_eq!(scalar(&op), Some(F::Float32), "ordinary output bias participates after projection");
}

#[test]
fn affine_projection_refuses_wrong_roles_and_preserves_empty_leading_dimensions() {
    let original = operation(Some(F::Float32), Some(F::Float32), Some(F::Float32));
    for (ordinal, replacement) in [
        (1, WorkspaceLayout::new(&[32, 9], WorkspaceDtype::Uint32).unwrap()),
        (2, floating_layout(&[32, 2], Some(F::Float32))),
        (3, WorkspaceLayout::new(&[32, 1], WorkspaceDtype::Uint32).unwrap()),
    ] {
        let mut changed = original.clone(); changed.inputs[ordinal] = replacement;
        assert_eq!(scalar(&changed), None);
    }
    let mut empty = original;
    empty.inputs[0] = floating_layout(&[1, 0, 64], Some(F::Float32));
    empty.outputs[0] = floating_layout(&[1, 0, 32], None);
    assert_eq!(scalar(&empty), Some(F::Float32));
    empty.outputs[0] = floating_layout(&[1, 2, 32], None);
    assert_eq!(scalar(&empty), None);
}

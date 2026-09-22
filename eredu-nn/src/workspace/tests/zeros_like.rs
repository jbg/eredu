use super::*;

#[test]
fn zeros_like_uses_the_exact_typed_constructor_without_a_value_dependency() {
    let floating = [
        (WorkspaceFloatingType::Float32, "zeros_f32"),
        (WorkspaceFloatingType::Float16, "zeros_f16"),
        (WorkspaceFloatingType::Bfloat16, "zeros_bf16"),
    ];
    for shape in [&[][..], &[0, 8][..], &[1, 8][..], &[3, 8][..]] {
        for (physical, name) in floating {
            let context = context();
            let input = WorkspaceTensor::existing(
                WorkspaceLayout::new(shape, WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(physical, false))),
                &context,
            )
            .unwrap();
            let output = input.zeros_like(&context).unwrap();
            assert_eq!(output.shape(), shape);
            let report = context.report(&[output]).unwrap();
            assert_eq!(report.operations.len(), 1);
            let operation = &report.operations[0];
            assert!(
                matches!(operation.kind, WorkspaceOperationKind::Elementwise(actual) if actual == name)
            );
            assert!(operation.inputs.is_empty());
            assert_eq!(operation.outputs[0].dtype(), WorkspaceDtype::Float32);
        }
        for (dtype, name) in [
            (WorkspaceDtype::Int32, "zeros_i32"),
            (WorkspaceDtype::Uint32, "zeros_u32"),
            (WorkspaceDtype::Uint8, "zeros_u8"),
            (WorkspaceDtype::Bool, "zeros_bool"),
        ] {
            let context = context();
            let input =
                WorkspaceTensor::existing(WorkspaceLayout::new(shape, dtype).unwrap(), &context)
                    .unwrap();
            let output = input.zeros_like(&context).unwrap();
            let report = context.report(&[output]).unwrap();
            assert_eq!(report.operations.len(), 1);
            assert!(
                matches!(report.operations[0].kind, WorkspaceOperationKind::Elementwise(actual) if actual == name)
            );
            assert!(report.operations[0].inputs.is_empty());
            assert_eq!(report.operations[0].outputs[0].dtype(), dtype);
        }
    }
}

#[test]
fn zeros_like_refuses_unknown_floating_precision_and_foreign_context() {
    let first = context();
    let second = context();
    let input = existing_f32(&[0, 8], &first).unwrap();
    assert!(input.zeros_like(&first).is_err());
    let typed = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[0, 8], WorkspaceDtype::Int32).unwrap(),
        &first,
    )
    .unwrap();
    assert!(typed.zeros_like(&second).is_err());
    assert!(first.report(&[]).unwrap().operations.is_empty());
    assert!(second.report(&[]).unwrap().operations.is_empty());
}

#[test]
fn unsigned_full_preserves_its_scalar_constructor_for_zero_and_nonzero_values() {
    for value in [0, 42, u32::MAX] {
        for shape in [&[][..], &[0, 4][..], &[1, 4][..]] {
            let context = context();
            let output = WorkspaceTensor::full_u32(value, shape, &context).unwrap();
            let report = context.report(&[output]).unwrap();
            assert_eq!(report.operations.len(), 1);
            let operation = &report.operations[0];
            assert!(matches!(
                operation.kind,
                WorkspaceOperationKind::Elementwise("full_u32")
            ));
            assert!(operation.inputs.is_empty());
            assert_eq!(operation.outputs[0].shape(), shape);
            assert_eq!(operation.outputs[0].dtype(), WorkspaceDtype::Uint32);
        }
    }
}

#[test]
fn signed_full_preserves_its_scalar_constructor_for_zero_and_nonzero_values() {
    for value in [i32::MIN, -42, 0, i32::MAX] {
        for shape in [&[][..], &[0, 4][..], &[1, 4][..]] {
            let context = context();
            let output = WorkspaceTensor::full_i32(value, shape, &context).unwrap();
            let report = context.report(&[output]).unwrap();
            assert_eq!(report.operations.len(), 1);
            let operation = &report.operations[0];
            assert!(matches!(
                operation.kind,
                WorkspaceOperationKind::Elementwise("full_i32")
            ));
            assert!(operation.inputs.is_empty());
            assert_eq!(operation.outputs[0].shape(), shape);
            assert_eq!(operation.outputs[0].dtype(), WorkspaceDtype::Int32);
        }
    }
}

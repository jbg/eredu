use super::*;

#[test]
fn row_scatter_preserves_prefix_geometry_and_source_broadcasting() {
    for (input, mask, source) in [
        (&[1, 17, 16][..], &[1, 17][..], &[4, 16][..]),
        (&[2, 3, 4][..], &[2, 3][..], &[2, 1][..]),
        (&[2, 3, 4][..], &[2][..], &[3, 1][..]),
        (&[2, 3][..], &[2, 3][..], &[4][..]),
        (&[2, 3][..], &[2][..], &[][..]),
        (&[2, 3][..], &[][..], &[2, 3][..]),
        (&[0, 3][..], &[0][..], &[0, 3][..]),
        (&[][..], &[][..], &[][..]),
    ] {
        let context = context();
        let input = existing_f32(input, &context).unwrap();
        let mask = WorkspaceTensor::existing(
            WorkspaceLayout::new(mask, WorkspaceDtype::Bool).unwrap(),
            &context,
        )
        .unwrap();
        let source = existing_i32(source, &context).unwrap();
        let result = input.masked_scatter(&mask, &source, &context).unwrap();
        assert_eq!(result.shape(), input.shape());
        assert_eq!(result.layout().dtype(), WorkspaceDtype::Float32);
        let report = context.report(&[result]).unwrap();
        assert_eq!(report.operations.len(), 1);
        let operation = &report.operations[0];
        assert!(matches!(
            operation.kind,
            WorkspaceOperationKind::Elementwise("masked_scatter")
        ));
        assert_eq!(operation.inputs[1].shape(), mask.shape());
        assert_eq!(operation.inputs[2].shape(), source.shape());
        assert_eq!(report.retained_bytes, Some(input.layout().bytes().unwrap()));
    }
}

#[test]
fn row_scatter_rejects_trailing_masks_and_incompatible_source_rows() {
    for (input, mask, source, dtype) in [
        (
            &[2, 3, 4][..],
            &[3, 4][..],
            &[2, 4][..],
            WorkspaceDtype::Bool,
        ),
        (
            &[2, 3, 4][..],
            &[1, 3][..],
            &[2, 4][..],
            WorkspaceDtype::Bool,
        ),
        (&[2, 3][..], &[2, 3, 1][..], &[1][..], WorkspaceDtype::Bool),
        (
            &[2, 3, 4][..],
            &[2, 3][..],
            &[2, 5][..],
            WorkspaceDtype::Bool,
        ),
        (
            &[2, 3, 4][..],
            &[2, 3][..],
            &[1, 2, 4][..],
            WorkspaceDtype::Bool,
        ),
        (&[2, 3][..], &[2][..], &[1, 3][..], WorkspaceDtype::Int32),
    ] {
        let context = context();
        let input = existing_f32(input, &context).unwrap();
        let mask = WorkspaceTensor::existing(WorkspaceLayout::new(mask, dtype).unwrap(), &context)
            .unwrap();
        let source = existing_f32(source, &context).unwrap();
        assert!(input.masked_scatter(&mask, &source, &context).is_err());
        assert!(context.report(&[]).unwrap().operations.is_empty());
    }
}

#[test]
fn row_scatter_geometry_does_not_grant_an_unknown_mechanism_a_storage_bound() {
    #[derive(Debug)]
    struct Unknown;
    impl WorkspaceMechanisms for Unknown {
        fn operation_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            Ok(None)
        }
    }
    let context = WorkspaceContext::new(Unknown);
    let input = existing_f32(&[1, 17, 16], &context).unwrap();
    let mask = WorkspaceTensor::existing(
        WorkspaceLayout::new(&[1, 17], WorkspaceDtype::Bool).unwrap(),
        &context,
    )
    .unwrap();
    let source = existing_f32(&[4, 16], &context).unwrap();
    let result = input.masked_scatter(&mask, &source, &context).unwrap();
    assert_eq!(context.report(&[result]).unwrap().total_bytes, None);
}

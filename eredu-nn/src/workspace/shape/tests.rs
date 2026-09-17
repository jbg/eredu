use super::*;
use crate::{
    workspace::{
        WorkspaceContext, WorkspaceDtype, WorkspaceLayout, WorkspaceLayoutView,
        WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound, WorkspaceTensor,
    },
    Tensor,
};

#[test]
fn borrowed_broadcast_preserves_scalars_zero_axes_and_high_rank() {
    for (left, right, expected) in [
        (&[][..], &[][..], &[][..]),
        (&[][..], &[2, 3][..], &[2, 3][..]),
        (&[2, 1, 7][..], &[3, 1][..], &[2, 3, 7][..]),
        (&[2, 0, 7][..], &[1, 7][..], &[2, 0, 7][..]),
    ] {
        let plan = WorkspaceBroadcastShape::new(left, right).unwrap();
        assert_eq!(plan.rank(), expected.len());
        let mut result = vec![-91; plan.rank()];
        plan.write_into(&mut result).unwrap();
        assert_eq!(result, expected);
        WorkspaceLayoutView::new(&result, WorkspaceDtype::Float32).unwrap();
    }
    let mut left = [1; 257];
    left[0] = 3;
    left[256] = 7;
    let right = [2, 1];
    let plan = WorkspaceBroadcastShape::new(&left, &right).unwrap();
    let mut result = [0; 257];
    plan.write_into(&mut result).unwrap();
    left[255] = 2;
    assert_eq!(result, left);
    assert_eq!(
        WorkspaceLayoutView::new(&result, WorkspaceDtype::Float32)
            .unwrap()
            .bytes(),
        Ok(168)
    );
}

#[test]
fn borrowed_matmul_and_ordinary_execution_share_vector_and_batch_geometry() {
    #[derive(Debug)]
    struct GeometryOnly;
    impl WorkspaceMechanisms for GeometryOnly {
        fn operation_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            Ok(None)
        }
    }
    let cases: &[(&[i32], &[i32], &[i32])] = &[
        (&[11], &[11], &[]),
        (&[11], &[11, 3], &[3]),
        (&[7, 11], &[11], &[7]),
        (&[2, 1, 7, 11], &[3, 11, 5], &[2, 3, 7, 5]),
        (&[11], &[2, 11, 5], &[2, 5]),
        (&[2, 7, 11], &[11], &[2, 7]),
        (&[2, 0, 11], &[1, 11, 5], &[2, 0, 5]),
    ];
    for &(left, right, expected) in cases {
        let plan = WorkspaceMatmulShape::new(left, right).unwrap();
        let mut result = vec![-91; plan.rank()];
        plan.write_into(&mut result).unwrap();
        assert_eq!(result, expected);
        let context = WorkspaceContext::new(GeometryOnly);
        let left = WorkspaceTensor::existing(
            WorkspaceLayout::new(left, WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        let right = WorkspaceTensor::existing(
            WorkspaceLayout::new(right, WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        assert_eq!(
            WorkspaceTensor::matmul(&left, &right, &context)
                .unwrap()
                .shape(),
            expected
        );
    }
}

#[test]
fn exact_shape_destinations_refuse_short_and_long_slices_without_writing() {
    let broadcast = WorkspaceBroadcastShape::new(&[2, 1, 7], &[3, 1]).unwrap();
    let matmul = WorkspaceMatmulShape::new(&[2, 7, 11], &[1, 11, 3]).unwrap();
    for length in [0, 2, 4, 9] {
        let mut result = vec![73; length];
        let expected = Err(WorkspaceShapeError::DestinationRank {
            expected: 3,
            actual: length,
        });
        assert_eq!(broadcast.write_into(&mut result), expected);
        assert_eq!(result, vec![73; length]);
        assert_eq!(matmul.write_into(&mut result), expected);
        assert_eq!(result, vec![73; length]);
    }
    let scalar = WorkspaceMatmulShape::new(&[7], &[7]).unwrap();
    scalar.write_into(&mut []).unwrap();
}

#[test]
fn borrowed_shape_refusals_preserve_matmul_validation_order() {
    assert_eq!(
        WorkspaceMatmulShape::new(&[], &[2, 3]),
        Err(WorkspaceShapeError::MatmulRank)
    );
    // Both contraction and batch widths disagree; contraction is checked first.
    assert_eq!(
        WorkspaceMatmulShape::new(&[2, 7, 11], &[3, 13, 5]),
        Err(WorkspaceShapeError::MatmulReduction)
    );
    assert_eq!(
        WorkspaceMatmulShape::new(&[2, 7, 11], &[3, 11, 5]),
        Err(WorkspaceShapeError::IncompatibleBroadcast)
    );
    assert_eq!(
        WorkspaceBroadcastShape::new(&[0, 7], &[2, 7]),
        Err(WorkspaceShapeError::IncompatibleBroadcast)
    );
    // Shape algebra never substitutes for layout validation.
    let plan = WorkspaceBroadcastShape::new(&[-2, 7], &[1, 7]).unwrap();
    let mut result = [0; 2];
    plan.write_into(&mut result).unwrap();
    assert_eq!(result, [-2, 7]);
    assert!(WorkspaceLayoutView::new(&result, WorkspaceDtype::Float32).is_err());
}

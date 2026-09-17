//! Borrowed geometry for ordinary and preallocated workspace inspection.
use super::{Error, WorkspaceDtype, WorkspaceRepresentation, workspace_overflow};

/// A shape calculation failed before creating owned inspection metadata.
/// This fixed value contains no diagnostic string or native error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum WorkspaceLayoutError {
    /// At least one dimension is negative, even if another is zero.
    #[error("negative workspace extent")]
    NegativeExtent,
    /// The nonempty shape's element count does not fit in `u64`.
    #[error("workspace element count overflow")]
    ElementCountOverflow,
    /// Elements fit, but their represented storage bytes do not.
    #[error("workspace tensor bytes overflow")]
    ByteCountOverflow,
}

impl WorkspaceLayoutError {
    pub(super) fn into_ordinary(self) -> Error {
        match self {
            Self::NegativeExtent => Error::backend("negative workspace extent"),
            Self::ElementCountOverflow => workspace_overflow("workspace element count overflow"),
            Self::ByteCountOverflow => workspace_overflow("workspace tensor bytes overflow"),
        }
    }
}

impl From<WorkspaceLayoutError> for Error {
    fn from(cause: WorkspaceLayoutError) -> Self {
        cause.into_ordinary()
    }
}

/// Validated tensor geometry borrowing its caller's dimensions.
///
/// Construction and queries allocate nothing. The caller retains the shape;
/// this view owns no tensor, storage account or native completion authority.
/// There is no rank limit beyond the supplied slice and checked arithmetic.
#[derive(Debug, Clone, Copy)]
pub struct WorkspaceLayoutView<'a> {
    pub(super) shape: &'a [i32],
    pub(super) dtype: WorkspaceDtype,
    pub(super) representation: Option<WorkspaceRepresentation>,
}

impl<'a> WorkspaceLayoutView<'a> {
    /// Checks all dimensions and represented byte arithmetic without copying.
    pub fn new(shape: &'a [i32], dtype: WorkspaceDtype) -> Result<Self, WorkspaceLayoutError> {
        let value = Self {
            shape,
            dtype,
            representation: None,
        };
        value.bytes()?;
        Ok(value)
    }

    /// The original dimensions, including zero extents.
    pub const fn shape(self) -> &'a [i32] {
        self.shape
    }

    /// Represented scalar type.
    pub const fn dtype(self) -> WorkspaceDtype {
        self.dtype
    }

    /// Logical elements; rank zero contains one scalar.
    pub fn elements(self) -> Result<u64, WorkspaceLayoutError> {
        joined_elements(self.shape, &[])
    }

    /// Logical bytes, excluding any mechanism-specific padding or copies.
    pub fn bytes(self) -> Result<u64, WorkspaceLayoutError> {
        Self::joined_bytes(self.shape, &[], self.dtype)
    }

    /// Checks the logical concatenation of two borrowed shape slices.
    ///
    /// This scalar query stores no dimensions and allocates no synthetic
    /// layout. Every dimension in both slices is validated before a zero
    /// suppresses product arithmetic. Two empty slices describe one scalar.
    pub fn joined_bytes(
        prefix: &[i32],
        tail: &[i32],
        dtype: WorkspaceDtype,
    ) -> Result<u64, WorkspaceLayoutError> {
        joined_elements(prefix, tail)?
            .checked_mul(dtype.bytes())
            .ok_or(WorkspaceLayoutError::ByteCountOverflow)
    }
}

fn joined_elements(prefix: &[i32], tail: &[i32]) -> Result<u64, WorkspaceLayoutError> {
    // Preserve the ordinary geometry rule across the complete logical shape:
    // a zero does not hide a negative, but any valid zero makes it empty.
    if prefix.iter().chain(tail).any(|n| *n < 0) {
        return Err(WorkspaceLayoutError::NegativeExtent);
    }
    if prefix.contains(&0) || tail.contains(&0) {
        return Ok(0);
    }
    prefix.iter().chain(tail).try_fold(1_u64, |size, n| {
        size.checked_mul(*n as u64)
            .ok_or(WorkspaceLayoutError::ElementCountOverflow)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Tensor,
        workspace::{
            WorkspaceContext, WorkspaceLayout, WorkspaceMechanisms, WorkspaceOperation,
            WorkspaceOperationBound, WorkspaceTensor,
        },
    };

    #[test]
    fn borrowed_layouts_match_owned_metadata_for_nonzero_scalar_empty_and_high_rank_shapes() {
        let high_rank = [1; 257];
        let shapes: &[&[i32]] = &[&[], &[2, 3, 5], &[7, 0, 11], &high_rank];
        for shape in shapes {
            for dtype in [
                WorkspaceDtype::Float32,
                WorkspaceDtype::Int32,
                WorkspaceDtype::Bool,
                WorkspaceDtype::Uint8,
                WorkspaceDtype::Uint32,
            ] {
                let view = WorkspaceLayoutView::new(shape, dtype).unwrap();
                let owned = WorkspaceLayout::new(shape, dtype).unwrap();
                assert_eq!(view.shape().as_ptr(), shape.as_ptr());
                assert_eq!(view.dtype(), dtype);
                assert_eq!(view.elements().unwrap(), owned.elements().unwrap());
                assert_eq!(view.bytes().unwrap(), owned.bytes().unwrap());
                assert_eq!(owned.as_view(), view);
                for split in 0..=shape.len() {
                    assert_eq!(
                        WorkspaceLayoutView::joined_bytes(&shape[..split], &shape[split..], dtype),
                        Ok(owned.bytes().unwrap()),
                    );
                }
            }
        }
        assert_eq!(
            WorkspaceLayoutView::new(&[], WorkspaceDtype::Float32)
                .unwrap()
                .bytes(),
            Ok(4)
        );
        assert_eq!(
            WorkspaceLayoutView::new(&[2, 3, 5], WorkspaceDtype::Uint32)
                .unwrap()
                .bytes(),
            Ok(120)
        );
    }

    #[test]
    fn geometry_refusals_preserve_negative_zero_and_distinct_overflow_precedence() {
        let max = i32::MAX;
        let cases: &[(&[i32], WorkspaceDtype, WorkspaceLayoutError)] = &[
            (
                &[0, -1],
                WorkspaceDtype::Float32,
                WorkspaceLayoutError::NegativeExtent,
            ),
            (
                &[-1, 0],
                WorkspaceDtype::Float32,
                WorkspaceLayoutError::NegativeExtent,
            ),
            (
                &[max, max, max, -1],
                WorkspaceDtype::Bool,
                WorkspaceLayoutError::NegativeExtent,
            ),
            (
                &[max, max, max],
                WorkspaceDtype::Bool,
                WorkspaceLayoutError::ElementCountOverflow,
            ),
            (
                &[max, max, 3],
                WorkspaceDtype::Float32,
                WorkspaceLayoutError::ByteCountOverflow,
            ),
        ];
        for (shape, dtype, expected) in cases {
            assert_eq!(WorkspaceLayoutView::new(shape, *dtype), Err(*expected));
            assert_eq!(
                WorkspaceLayout::new(shape, *dtype).unwrap_err().to_string(),
                expected.into_ordinary().to_string()
            );
        }
        assert_eq!(
            WorkspaceLayoutView::joined_bytes(&[max, max, max], &[0], WorkspaceDtype::Float32),
            Ok(0),
        );
        assert_eq!(
            WorkspaceLayoutView::joined_bytes(&[0, max, max], &[-1], WorkspaceDtype::Bool),
            Err(WorkspaceLayoutError::NegativeExtent),
        );
        assert_eq!(
            WorkspaceLayoutView::joined_bytes(&[], &[], WorkspaceDtype::Float32),
            Ok(4)
        );
        for shape in [&[0, max, max, max][..], &[max, max, max, 0][..]] {
            assert_eq!(
                WorkspaceLayoutView::new(shape, WorkspaceDtype::Float32)
                    .unwrap()
                    .bytes(),
                Ok(0)
            );
        }
    }

    #[test]
    fn ordinary_batched_matmul_exposes_borrowed_result_geometry() {
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
        let context = WorkspaceContext::new(GeometryOnly);
        let left = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[2, 7, 11], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        let right = WorkspaceTensor::existing(
            WorkspaceLayout::new(&[1, 11, 3], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        let tensor = WorkspaceTensor::matmul(&left, &right, &context).unwrap();
        let view = tensor.layout().as_view();
        assert_eq!(view.shape(), &[2, 7, 3]);
        assert_eq!(view.elements(), Ok(42));
        assert_eq!(view.bytes(), Ok(168));
        assert_eq!(view.shape().as_ptr(), tensor.layout().shape().as_ptr());
    }
}

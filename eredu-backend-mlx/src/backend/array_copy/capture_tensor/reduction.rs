//! Exact private static selection constructor shared by typed reductions.
use super::*;
impl Selection {
    /// Fixed shared constructor call/result representations, reused for each axis.
    pub(crate) fn reduction_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<Self>(),
            size_of::<Result<Self, CaptureTensorNativeError>>(),
            size_of::<(&[usize], &[usize], &[u64], &[u64], &[u64])>(),
            size_of::<(usize, std::ops::Range<usize>)>(),
            size_of::<Result<i32, std::num::TryFromIntError>>(),
            size_of::<Option<i32>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }

    pub(super) fn reduction(
        source_shape: &[usize],
        shape: &[usize],
        starts: &[u64],
        ends: &[u64],
        strides: &[u64],
    ) -> Result<Self, CaptureTensorNativeError> {
        let rank = source_shape.len();
        if [shape.len(), starts.len(), ends.len(), strides.len()]
            .iter()
            .any(|n| *n != rank)
        {
            return Err(CaptureTensorNativeError::ShapeMismatch);
        }
        if rank > 32 {
            return Err(CaptureTensorNativeError::ShapeMismatch);
        }
        let mut selection = Selection {
            source_shape: [0; 32],
            starts: [0; 32],
            ends: [0; 32],
            strides: [1; 32],
            selected_shape: [0; 32],
            rank,
            preview: None,
            cast_f32: false,
            read_f32: false,
        };
        for axis in 0..rank {
            selection.source_shape[axis] = i32::try_from(source_shape[axis])
                .map_err(|_| CaptureTensorNativeError::GeometryOverflow)?;
            selection.starts[axis] = i32::try_from(starts[axis])
                .map_err(|_| CaptureTensorNativeError::GeometryOverflow)?;
            selection.ends[axis] = i32::try_from(ends[axis])
                .map_err(|_| CaptureTensorNativeError::GeometryOverflow)?;
            selection.strides[axis] = i32::try_from(strides[axis])
                .map_err(|_| CaptureTensorNativeError::GeometryOverflow)?;
            selection.selected_shape[axis] = i32::try_from(shape[axis])
                .map_err(|_| CaptureTensorNativeError::GeometryOverflow)?;
            (selection.ends[axis] - selection.starts[axis])
                .checked_add(selection.strides[axis] - 1)
                .ok_or(CaptureTensorNativeError::GeometryOverflow)?;
        }
        Ok(selection)
    }
}

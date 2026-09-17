//! Physical layout evidence from the same completed native descriptor loan.
use super::*;
use safemlx::OwnedArrayDescriptorLoan;

/// No allocation, native execution, shape guessing or additional backing credit.
/// The descriptor retains the source and runtime guard through every stride read.
pub(super) fn from_descriptor(
    descriptor: &OwnedArrayDescriptorLoan<'_>,
) -> Option<WorkspaceRepresentation> {
    let representation = preparation::projected_representation(descriptor.facts().dtype())?;
    let mut representation = WorkspaceRepresentation::new(
        representation.dtype(),
        descriptor.row_contiguous() == Some(true),
    );
    let Some(strides) = descriptor.completed_strides() else {
        return Some(representation);
    };
    let shape = descriptor.shape();
    if strides.len() != shape.len() {
        return Some(representation);
    }
    representation = representation.with_last_axis_contiguous(
        shape.is_empty() || shape[shape.len() - 1] <= 1 || strides[strides.len() - 1] == 1,
    );
    if representation.row_contiguous() || shape.len() > 8 {
        return Some(representation);
    }
    // Singleton strides are immaterial. Keep those axes first in logical order,
    // then insertion-sort nonunit axes by decreasing actual positive stride.
    let mut axes = [0usize; 8];
    let mut count = 0;
    for axis in 0..shape.len() {
        if shape[axis] <= 0 {
            return Some(representation);
        }
        if shape[axis] == 1 {
            axes[count] = axis;
            count += 1;
        }
    }
    let singleton_count = count;
    for axis in 0..shape.len() {
        if shape[axis] == 1 {
            continue;
        }
        if strides[axis] <= 0 {
            return Some(representation);
        }
        let mut position = count;
        while position > singleton_count && strides[axes[position - 1]] < strides[axis] {
            axes[position] = axes[position - 1];
            position -= 1;
        }
        axes[position] = axis;
        count += 1;
    }
    let mut expected = 1i64;
    for position in (singleton_count..count).rev() {
        let axis = axes[position];
        if strides[axis] != expected {
            return Some(representation);
        }
        let Some(next) = expected.checked_mul(i64::from(shape[axis])) else {
            return Some(representation);
        };
        expected = next;
    }
    Some(
        representation
            .with_dense_axis_order(&axes[..count])
            .unwrap_or(representation),
    )
}

/// Fixed maximum frame, reserved by the common import query before inspection.
pub(super) fn control_bytes() -> Option<usize> {
    let parts = [
        size_of::<&OwnedArrayDescriptorLoan<'_>>(),
        size_of::<WorkspaceRepresentation>() * 2,
        size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<Option<&[i64]>>(),
        size_of::<(&[i32], &[i64])>(),
        size_of::<[usize; 8]>(),
        size_of::<[usize; 6]>(),
        size_of::<[i64; 2]>(),
        size_of::<Option<i64>>(),
        size_of::<std::ops::Range<usize>>() * 2,
        size_of::<std::iter::Rev<std::ops::Range<usize>>>(),
        // The compact neutral encoder uses only these scalar controls.
        size_of::<(&[usize], u16, u32, usize, usize)>(),
        size_of::<std::iter::Enumerate<std::slice::Iter<'_, usize>>>(),
    ];
    parts
        .into_iter()
        .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
}

//! Fixed destinations for the same contiguous partition equations.
use super::*;
use std::ops::Range;
mod construction;
pub use construction::CaptureContiguousProjectionPlan;
/// Fixed geometry refusal; no declaration, tensor, or allocation authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CaptureContiguousProjectionError {
    /// Shape, rank, or partition axis differs from the global selection.
    #[error("partition capture axis/rank differs from global geometry")]
    Axes,
    /// An input slice is not the exact normalized global selection.
    #[error("invalid global partition capture slice")]
    Slice,
    /// The actual contiguous source range is outside the global axis.
    #[error("contiguous capture range exceeds global geometry")]
    Range,
    /// A preallocated output does not have the exact source rank.
    #[error("contiguous capture destination rank differs")]
    Destination,
    /// The actual nonempty source requires one fragment beyond the selected cap.
    #[error("partition capture fragment count exceeds its bound")]
    Fragments,
    /// An element extent or arithmetic product overflows.
    #[error("capture geometry overflow")]
    Overflow,
}
impl CaptureContiguousProjectionError {
    pub(super) fn legacy(self) -> CaptureError {
        match self {
            Self::Axes => CaptureError::Invalid(
                "partition capture axis/rank differs from global geometry".into(),
            ),
            Self::Slice => CaptureError::Invalid("invalid global partition capture slice".into()),
            Self::Range => {
                CaptureError::Invalid("contiguous capture range exceeds global geometry".into())
            }
            Self::Destination => {
                CaptureError::Invalid("contiguous capture destination rank differs".into())
            }
            Self::Fragments => CaptureError::Invalid("partition capture fragment count exceeds its bound".into()),
            Self::Overflow => CaptureError::Overflow,
        }
    }
}
pub(super) fn validate(
    global: &[u64],
    slice: &ResolvedCaptureSlice,
    axis: usize,
    count: u64,
) -> Result<(), CaptureContiguousProjectionError> {
    use CaptureContiguousProjectionError as E;
    let rank = global.len();
    if rank == 0
        || rank > 32
        || axis >= rank
        || [&slice.starts, &slice.ends, &slice.strides, &slice.shape]
            .iter()
            .any(|v| v.len() != rank)
        || global[axis] != count
    {
        return Err(E::Axes);
    }
    elements(global).map_err(|_| E::Overflow)?;
    for (i, &extent) in global.iter().enumerate() {
        let (start, end, stride) = (slice.starts[i], slice.ends[i], slice.strides[i]);
        if start > end
            || end > extent
            || stride == 0
            || (end - start).div_ceil(stride) != slice.shape[i]
        {
            return Err(E::Slice);
        }
    }
    Ok(())
}
pub(super) struct Span {
    pub(super) local_first: u64,
    pub(super) local_last: u64,
    pub(super) stride: u64,
    pub(super) first: u64,
    pub(super) last: u64,
}
pub(super) fn span(slice: &ResolvedCaptureSlice, axis: usize, range: Range<u64>) -> Option<Span> {
    let (start, end, stride) = (slice.starts[axis], slice.ends[axis], slice.strides[axis]);
    let lower = range.start.max(start);
    let upper = range.end.min(end);
    if lower >= upper {
        return None;
    }
    let first = (lower - start).div_ceil(stride);
    if first >= slice.shape[axis] {
        return None;
    }
    let last = (upper - 1 - start) / stride;
    if first > last {
        return None;
    }
    // The exact normalized selection proves both products remain below end.
    Some(Span {
        local_first: start + first * stride - range.start,
        local_last: start + last * stride - range.start,
        stride,
        first,
        last,
    })
}
impl CaptureSlicePartition {
    /// Project a globally anchored selection onto one contiguous physical range.
    /// Writes only caller-owned exact-rank buffers. False means no overlap; the
    /// destinations are untouched and there is no native update to construct.
    pub fn contiguous_fragment_into(
        global: &[u64],
        slice: &ResolvedCaptureSlice,
        axis: usize,
        range: Range<u64>,
        local: &mut ResolvedCaptureSlice,
        destination: &mut ResolvedCaptureSlice,
    ) -> Result<bool, CaptureContiguousProjectionError> {
        Self::contiguous_axes_into(global, slice, &[(axis, range)], local, destination)
    }
    /// Project the same normalized selection through distinct physical axis
    /// ranges. Payload destinations remain ordinals in the original selection.
    /// False leaves both fixed destinations untouched, including empty prefixes.
    pub fn contiguous_axes_into(
        global: &[u64],
        slice: &ResolvedCaptureSlice,
        ranges: &[(usize, Range<u64>)],
        local: &mut ResolvedCaptureSlice,
        destination: &mut ResolvedCaptureSlice,
    ) -> Result<bool, CaptureContiguousProjectionError> {
        use CaptureContiguousProjectionError as E;
        let &(first, _) = ranges.first().ok_or(E::Axes)?;
        let count = global.get(first).copied().ok_or(E::Axes)?;
        validate(global, slice, first, count)?;
        let rank = global.len();
        let mut seen = [false; 32];
        for (axis, range) in ranges {
            let count = *global.get(*axis).ok_or(E::Axes)?;
            if seen[*axis] { return Err(E::Axes); }
            seen[*axis] = true;
            if range.start > range.end || range.end > count { return Err(E::Range); }
        }
        if [
            &local.starts, &local.ends, &local.strides, &local.shape,
            &destination.starts, &destination.ends, &destination.strides, &destination.shape,
        ].iter().any(|v| v.len() != rank) { return Err(E::Destination); }
        if elements(&slice.shape).map_err(|_| E::Overflow)? == 0 { return Ok(false); }
        for (axis, range) in ranges {
            let Some(projected) = span(slice, *axis, range.clone()) else { return Ok(false); };
            projected.local_last.checked_add(1).ok_or(E::Overflow)?;
            projected.last.checked_add(1).ok_or(E::Overflow)?;
        }
        local.starts.copy_from_slice(&slice.starts);
        local.ends.copy_from_slice(&slice.ends);
        local.strides.copy_from_slice(&slice.strides);
        local.shape.copy_from_slice(&slice.shape);
        destination.starts.fill(0);
        destination.ends.copy_from_slice(&slice.shape);
        destination.strides.fill(1);
        destination.shape.copy_from_slice(&slice.shape);
        for (axis, range) in ranges {
            // The unchanged selection and ranges passed the complete preflight.
            let projected = span(slice, *axis, range.clone()).ok_or(E::Range)?;
            let length = (projected.local_last - projected.local_first) / projected.stride + 1;
            local.starts[*axis] = projected.local_first;
            local.ends[*axis] = projected.local_last + 1;
            local.strides[*axis] = projected.stride;
            local.shape[*axis] = length;
            destination.starts[*axis] = projected.first;
            destination.ends[*axis] = projected.last + 1;
            destination.shape[*axis] = length;
        }
        Ok(true)
    }
    /// Fixed validation and projection controls; output Vec backing must be paid
    /// by its actual caller before entering the shared worker.
    pub fn contiguous_projection_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let parts = [
            size_of::<Span>(),
            size_of::<[bool; 32]>(),
            size_of::<[(usize, Range<u64>); 1]>(),
            size_of::<(&[u64], &ResolvedCaptureSlice, &[(usize, Range<u64>)],
                &mut ResolvedCaptureSlice, &mut ResolvedCaptureSlice)>(),
            size_of::<Option<Span>>(),
            size_of::<[u64; 12]>(),
            size_of::<Range<u64>>(),
            size_of::<[&Vec<u64>; 8]>(),
            size_of::<(
                &[u64],
                &ResolvedCaptureSlice,
                usize,
                Range<u64>,
                &mut ResolvedCaptureSlice,
                &mut ResolvedCaptureSlice,
            )>(),
            size_of::<Result<bool, CaptureContiguousProjectionError>>(),
            size_of::<CaptureContiguousProjectionError>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

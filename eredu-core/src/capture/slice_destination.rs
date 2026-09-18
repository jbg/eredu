//! Existing half-open slice worker with caller-provided exact destinations.
use super::*;

/// Allocation-free refusal from borrowed slice resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CaptureSliceDestinationError {
    /// Actual rank differs from the declaration.
    #[error("runtime tensor rank differs from catalog")]
    Rank,
    /// A named source axis is absent.
    #[error("unknown selected axis at index {index}")]
    Axis { index: usize },
    /// A selected range exceeds the actual source.
    #[error("selected range at index {index} exceeds runtime extent")]
    Range { index: usize },
    /// The supplied fixed destinations do not have the source rank.
    #[error("slice destination rank differs from source")]
    Destination,
    /// The selected element count overflowed.
    #[error("slice geometry overflow")]
    Overflow,
}
impl CaptureSliceDestinationError {
    pub(crate) fn legacy(self, slices: &[CaptureSlice]) -> CaptureError {
        self.legacy_with(slices,super::admission::allocation::Allocation(None))
    }
    pub(crate) fn legacy_with(self, slices: &[CaptureSlice],
        allocation: super::admission::allocation::Allocation<'_>) -> CaptureError {
        let result = (|| -> Result<CaptureError,CaptureError> { Ok(
        match self {
            Self::Rank => CaptureError::Invalid(allocation.text("runtime tensor rank differs from catalog")?),
            Self::Axis { index } => {
                CaptureError::Invalid(allocation.format(format_args!("unknown axis {}", slices[index].axis))?)
            }
            Self::Range { index } => CaptureError::Invalid(allocation.format(format_args!(
                "slice {} exceeds runtime extent",
                slices[index].axis
            ))?),
            Self::Destination => {
                CaptureError::Invalid(allocation.text("slice destination rank differs from source")?)
            }
            Self::Overflow => CaptureError::Overflow,
        }
        ) })();
        result.unwrap_or_else(|cause|cause)

    }
}
/// Resolves into four existing rank-sized slices. It constructs no payload or
/// diagnostic String; the caller retains and funds the destinations.
pub fn resolve_slice_into(
    axes: Option<&[crate::TensorAxis]>,
    slices: &[CaptureSlice],
    shape: &[u64],
    starts: &mut [u64],
    ends: &mut [u64],
    strides: &mut [u64],
    selected: &mut [u64],
) -> Result<(), CaptureSliceDestinationError> {
    use CaptureSliceDestinationError as E;
    if axes.is_some_and(|a| a.len() != shape.len()) {
        return Err(E::Rank);
    }
    if [starts.len(), ends.len(), strides.len(), selected.len()]
        .iter()
        .any(|n| *n != shape.len())
    {
        return Err(E::Destination);
    }
    starts.fill(0);
    ends.copy_from_slice(shape);
    strides.fill(1);
    selected.copy_from_slice(shape);
    for (index, slice) in slices.iter().enumerate() {
        let axis = axes
            .and_then(|a| a.iter().position(|a| a.name == slice.axis))
            .ok_or(E::Axis { index })?;
        if slice.stride == 0 || slice.start > slice.end || slice.end > shape[axis] {
            return Err(E::Range { index });
        }
        starts[axis] = slice.start;
        ends[axis] = slice.end;
        strides[axis] = slice.stride;
        selected[axis] = (slice.end - slice.start).div_ceil(slice.stride);
    }
    elements(selected).map_err(|_| E::Overflow)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prepared_slice_destination_preserves_strides_and_zero_extent_overflow() {
        let axes=[crate::TensorAxis {name:"vocabulary".into(),dimension:crate::SymbolicDimension::Known(9)}];
        let slices=[CaptureSlice {axis:"vocabulary".into(),start:1,end:9,stride:3}];
        let (mut starts,mut ends,mut strides,mut selected)=([0],[0],[0],[0]);
        resolve_slice_into(Some(&axes),&slices,&[9],&mut starts,&mut ends,&mut strides,&mut selected).unwrap();
        assert_eq!((starts,ends,strides,selected),([1],[9],[3],[3]));
        let (mut starts,mut ends,mut strides,mut selected)=([0;3],[0;3],[0;3],[0;3]);
        assert_eq!(resolve_slice_into(None,&[],&[0,u64::MAX,2],&mut starts,&mut ends,&mut strides,&mut selected),
            Err(CaptureSliceDestinationError::Overflow));
    }
}

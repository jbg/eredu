//! Borrowed logical-row projection shared by ordinary and prepared consumers.
use super::*;

/// Fixed refusal from an exact physical-to-logical invocation projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CaptureWindowSourceError {
    /// The physical/logical invocation or actual axes differ.
    #[error(transparent)]
    Axes(#[from] CaptureAxisError),
    /// The single-lane physical span is outside the logical row extent.
    #[error("invalid single-lane invocation window")]
    Window,
    /// Window projection requires retained axes.
    #[error("window capture requires declared row axes")]
    MissingAxes,
    /// The source has no Sequence or TokenRows declaration.
    #[error("window capture has no declared row axis")]
    MissingRows,
    /// More than one declared row interpretation is ambiguous.
    #[error("window capture requires one declared row axis")]
    MultipleRows,
    /// The caller's exact-rank destination differs from the actual source.
    #[error("window source destination rank differs")]
    Destination,
}
impl CaptureWindowSourceError {
    pub(super) fn legacy(self, point: Option<&ObservationPoint>) -> CaptureError {
        match self {
            Self::Axes(cause) => cause.legacy(point),
            Self::Window => CaptureError::Invalid("invalid single-lane invocation window".into()),
            Self::MissingAxes => {
                CaptureError::Unsupported("window capture requires declared row axes".into())
            }
            Self::MissingRows => {
                CaptureError::Unsupported("window capture has no declared row axis".into())
            }
            Self::MultipleRows => {
                CaptureError::Unsupported("window capture requires one declared row axis".into())
            }
            Self::Destination => {
                CaptureError::Invalid("window source destination rank differs".into())
            }
        }
    }
}
pub(super) struct SourceAxes {
    pub(super) logical: CaptureInvocationShape,
    pub(super) axis: usize,
}
impl CaptureInvocationWindow {
    /// Same ordinary span validation without allocating an error diagnostic.
    pub fn validate_fixed(
        self,
        physical: CaptureInvocationShape,
    ) -> Result<CaptureInvocationShape, CaptureWindowSourceError> {
        physical.validate_fixed()?;
        if physical.batch != 1
            || self.logical_sequence == 0
            || self
                .start
                .checked_add(physical.sequence)
                .is_none_or(|end| end > self.logical_sequence)
        {
            return Err(CaptureWindowSourceError::Window);
        }
        let logical = CaptureInvocationShape {
            sequence: self.logical_sequence,
            ..physical
        };
        logical.validate_fixed()?;
        Ok(logical)
    }
    pub(super) fn source_axes_plan(
        self,
        physical: CaptureInvocationShape,
        axes: Option<&[crate::TensorAxis]>,
        actual: &[u64],
    ) -> Result<SourceAxes, CaptureWindowSourceError> {
        let logical = self.validate_fixed(physical)?;
        physical.validate_actual_axes(axes, actual)?;
        let axes = axes.ok_or(CaptureWindowSourceError::MissingAxes)?;
        let mut rows = axes.iter().enumerate().filter(|(_, axis)| {
            matches!(
                axis.dimension,
                SymbolicDimension::Sequence | SymbolicDimension::TokenRows
            )
        });
        let axis = rows
            .next()
            .map(|(i, _)| i)
            .ok_or(CaptureWindowSourceError::MissingRows)?;
        if rows.next().is_some() {
            return Err(CaptureWindowSourceError::MultipleRows);
        }
        Ok(SourceAxes { logical, axis })
    }
    /// Write the logical source extents into a previously paid exact-rank
    /// destination. Axis interpretation and validation share the ordinary worker;
    /// this descriptor supplies no source, quota, model or completion authority.
    pub fn source_axes_into(
        self,
        physical: CaptureInvocationShape,
        axes: Option<&[crate::TensorAxis]>,
        actual: &[u64],
        global: &mut [u64],
    ) -> Result<(CaptureInvocationShape, usize), CaptureWindowSourceError> {
        let plan = self.source_axes_plan(physical, axes, actual)?;
        if global.len() != actual.len() {
            return Err(CaptureWindowSourceError::Destination);
        }
        global.copy_from_slice(actual);
        global[plan.axis] = plan.logical.sequence;
        plan.logical.validate_actual_axes(axes, global)?;
        Ok((plan.logical, plan.axis))
    }
    /// Actual fixed source/window/axis validation frames. Destination backing is
    /// separate and comes from its consumer's counted rank allocation.
    pub fn source_axes_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let parts = [
            size_of::<Self>(),
            size_of::<[CaptureInvocationShape; 2]>(),
            size_of::<SourceAxes>(),
            size_of::<Option<usize>>(),
            size_of::<(
                Self,
                CaptureInvocationShape,
                Option<&[crate::TensorAxis]>,
                &[u64],
                &mut [u64],
            )>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'static, crate::TensorAxis>>>(),
            size_of::<Result<SourceAxes, CaptureWindowSourceError>>(),
            size_of::<Result<(CaptureInvocationShape, usize), CaptureWindowSourceError>>(),
            size_of::<CaptureWindowSourceError>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn buffer() -> ResolvedCaptureSlice {
        ResolvedCaptureSlice {
            starts: vec![99; 3],
            ends: vec![99; 3],
            strides: vec![99; 3],
            shape: vec![99; 3],
        }
    }
    #[test]
    fn prepared_windows_preserve_global_stride_payload_destinations_and_no_overlap() {
        let axes = [
            crate::TensorAxis {
                name: "batch".into(),
                dimension: SymbolicDimension::Batch,
            },
            crate::TensorAxis {
                name: "sequence".into(),
                dimension: SymbolicDimension::Sequence,
            },
            crate::TensorAxis {
                name: "hidden".into(),
                dimension: SymbolicDimension::Known(4),
            },
        ];
        let slice = ResolvedCaptureSlice {
            starts: vec![0, 1, 1],
            ends: vec![1, 5, 4],
            strides: vec![1, 2, 2],
            shape: vec![1, 2, 2],
        };
        let mut global = [0; 3];
        let mut local = buffer();
        let mut destination = buffer();
        let pointers = (local.starts.as_ptr(), destination.starts.as_ptr());
        let mut local_rows = Vec::new();
        let mut payload_rows = Vec::new();
        for range in [0..2, 2..4, 4..5] {
            let physical = CaptureInvocationShape {
                batch: 1,
                sequence: range.end - range.start,
                context: None,
            };
            let window = CaptureInvocationWindow {
                logical_sequence: 5,
                start: range.start,
            };
            let (logical, axis) = window
                .source_axes_into(
                    physical,
                    Some(&axes),
                    &[1, physical.sequence, 4],
                    &mut global,
                )
                .unwrap();
            assert_eq!(logical.sequence, 5);
            assert_eq!(axis, 1);
            assert_eq!(global, [1, 5, 4]);
            let previous = (local.clone(), destination.clone());
            let found = CaptureSlicePartition::contiguous_fragment_into(
                &global,
                &slice,
                axis,
                range.clone(),
                &mut local,
                &mut destination,
            )
            .unwrap();
            let ordinary = CaptureSlicePartition::new(
                &global,
                &slice,
                axis,
                &crate::component::ComponentCoordinateMap::range(
                    5,
                    range.start as usize..range.end as usize,
                )
                .unwrap(),
                1,
            )
            .unwrap();
            assert_eq!(found, !ordinary.fragments().is_empty());
            if found {
                assert_eq!(&local, ordinary.fragments()[0].local());
                assert_eq!(&destination, ordinary.fragments()[0].destination());
                local_rows.push(range.start + local.starts[1]);
                payload_rows.push(destination.starts[1]);
            } else {
                assert_eq!((local.clone(), destination.clone()), previous);
            }
            assert_eq!(
                (local.starts.as_ptr(), destination.starts.as_ptr()),
                pointers
            );
        }
        assert_eq!(local_rows, [1, 3]);
        assert_eq!(payload_rows, [0, 1]);
        let before = (local.clone(), destination.clone());
        assert_eq!(
            CaptureSlicePartition::contiguous_fragment_into(
                &global,
                &slice,
                1,
                4..6,
                &mut local,
                &mut destination
            ),
            Err(CaptureContiguousProjectionError::Range)
        );
        assert_eq!((local, destination), before);
        assert!(CaptureInvocationWindow::source_axes_control_bytes().unwrap() > 0);
        assert!(CaptureSlicePartition::contiguous_projection_control_bytes().unwrap() > 0);
    }
}

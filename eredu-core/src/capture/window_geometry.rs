//! Fixed geometry shared by existing spatial and temporal capture consumers.
use super::*;
fn invalid(message: &str) -> CaptureError {
    CaptureError::Invalid(message.into())
}
/// Allocation-free validation failure from the shared declared-window worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CaptureWindowError {
    /// A declared or physical window violates the fixed geometry contract.
    #[error("{0}")]
    Invalid(&'static str),
    /// The selected declaration cannot be represented by this window mechanism.
    #[error("{0}")]
    Unsupported(&'static str),
    /// Checked window arithmetic cannot be represented.
    #[error("capture window arithmetic overflow")]
    Overflow,
}
impl From<CaptureWindowError> for CaptureError {
    fn from(value: CaptureWindowError) -> Self {
        match value {
            CaptureWindowError::Invalid(message) => Self::Invalid(message.into()),
            CaptureWindowError::Unsupported(message) => Self::Unsupported(message.into()),
            CaptureWindowError::Overflow => Self::Overflow,
        }
    }
}
fn invalid_fixed(message: &'static str) -> CaptureWindowError {
    CaptureWindowError::Invalid(message)
}
fn window_add(a: u64, b: u64) -> Result<u64, CaptureWindowError> {
    add(a, b).map_err(|_| CaptureWindowError::Overflow)
}
fn window_mul(a: u64, b: u64) -> Result<u64, CaptureWindowError> {
    mul(a, b).map_err(|_| CaptureWindowError::Overflow)
}
fn window_elements(shape: &[u64]) -> Result<u64, CaptureWindowError> {
    elements(shape).map_err(|_| CaptureWindowError::Overflow)
}
/// Read-only declared tensor window geometry. It is not a source witness or grant.
#[derive(Debug, Clone, Copy)]
pub struct CaptureWindowGeometry {
    rank: usize,
    axis: usize,
    source: [u64; 32],
    selected: [u64; 32],
    start: u64,
    end: u64,
    stride: u64,
    elements: u64,
    // Only this actual spatial projection overrides the original named slice.
    partition: Option<(usize, u64, u64)>,
}
impl CaptureWindowGeometry {
    /// Number of actual declared axes.
    pub fn rank(&self) -> usize {
        self.rank
    }
    /// Single declared Sequence/TokenRows axis. No causal support is inferred.
    pub fn axis(&self) -> usize {
        self.axis
    }
    /// Fixed source extents; only the first rank entries are meaningful.
    pub fn source(&self) -> &[u64; 32] {
        &self.source
    }
    /// Fixed selected extents before Preview; only rank entries are meaningful.
    pub fn selected(&self) -> &[u64; 32] {
        &self.selected
    }
    /// Original selected row start, anchoring every fragment's stride.
    pub fn start(&self) -> u64 {
        self.start
    }
    /// Original selected row stride.
    pub fn stride(&self) -> u64 {
        self.stride
    }
    /// Total selected elements before any prefix truncation.
    pub fn elements(&self) -> u64 {
        self.elements
    }

    /// Resolve declared axes with explicit invocation extents. This utility
    /// supplies no source identity, execution support, funding or completion.
    pub fn new(
        point: &crate::ObservationPoint,
        selection: &CaptureSelection,
        batch: u64,
        rows: u64,
    ) -> Result<Self, CaptureError> {
        Self::new_fixed(point, selection, batch, rows).map_err(Into::into)
    }
    /// Resolve through the same worker while retaining a fixed error value.
    /// This supplies no native, source or allocation authority.
    pub fn new_fixed(
        point: &crate::ObservationPoint,
        selection: &CaptureSelection,
        batch: u64,
        rows: u64,
    ) -> Result<Self, CaptureWindowError> {
        Self::resolve(point, selection, batch, rows, None)
    }
    // An invocation already resolved every axis through its exact admission.
    // Non-row dynamic dimensions keep that actual physical extent; the public
    // aggregate constructor still requires its existing fixed non-row geometry.
    pub(super) fn from_resolved(
        point: &crate::ObservationPoint,
        selection: &CaptureSelection,
        batch: u64,
        rows: u64,
        resolved: &[usize],
    ) -> Result<Self, CaptureError> {
        Self::resolve(point, selection, batch, rows, Some(resolved)).map_err(Into::into)
    }
    fn resolve(
        point: &crate::ObservationPoint,
        selection: &CaptureSelection,
        batch: u64,
        rows: u64,
        resolved: Option<&[usize]>,
    ) -> Result<Self, CaptureWindowError> {
        // A sparse invocation already supplies its exact token/slot/unit
        // geometry. Share physical row intersection without admitting sparse
        // tensors to the unrelated logical reduction constructor.
        let sparse_invocation = resolved.is_some()
            && matches!(
                point.value_type,
                crate::ObservationValueType::RoutedUnits { .. }
            )
            && matches!(selection.transform, CaptureTransform::RoutedUnits);
        if point.value_type != crate::ObservationValueType::Tensor && !sparse_invocation {
            return Err(CaptureWindowError::Unsupported(
                "reduction windows require ordinary tensor hooks",
            ));
        }
        let axes = point
            .axes
            .as_ref()
            .ok_or_else(|| invalid_fixed("reduction window has no declared axes"))?;
        if axes.len() > 32 {
            return Err(CaptureWindowError::Unsupported(
                "reduction window rank exceeds 32",
            ));
        }
        let mut out = Self {
            rank: axes.len(),
            axis: usize::MAX,
            source: [0; 32],
            selected: [0; 32],
            start: 0,
            end: rows,
            stride: 1,
            elements: 0,
            partition: None,
        };
        for (i, axis) in axes.iter().enumerate() {
            use crate::SymbolicDimension as D;
            let extent = match axis.dimension {
                D::Sequence | D::TokenRows => {
                    if out.axis != usize::MAX {
                        return Err(invalid_fixed("reduction window has multiple row axes"));
                    }
                    out.axis = i;
                    rows
                }
                D::Batch => batch,
                D::Known(n) => u64::try_from(n).map_err(|_| CaptureWindowError::Overflow)?,
                _ => match resolved.and_then(|shape| shape.get(i)) {
                    Some(&n) => u64::try_from(n).map_err(|_| CaptureWindowError::Overflow)?,
                    None => {
                        return Err(CaptureWindowError::Unsupported(
                            "reduction windows require fixed non-row axes",
                        ));
                    }
                },
            };
            if resolved.is_some_and(|shape| {
                shape.len() != axes.len()
                    || shape.get(i).copied().and_then(|n| u64::try_from(n).ok()) != Some(extent)
            }) {
                return Err(invalid_fixed("resolved invocation window axes changed"));
            }
            out.source[i] = extent;
            out.selected[i] = extent;
            if let Some(slice) = selection.slices.iter().find(|s| s.axis == axis.name) {
                if slice.stride == 0 || slice.start > slice.end || slice.end > extent {
                    return Err(invalid_fixed(
                        "logical reduction slice exceeds original geometry",
                    ));
                }
                out.selected[i] = (slice.end - slice.start).div_ceil(slice.stride);
                if out.axis == i {
                    out.start = slice.start;
                    out.end = slice.end;
                    out.stride = slice.stride;
                }
            }
        }
        if out.axis == usize::MAX {
            return Err(CaptureWindowError::Unsupported(
                "reduction window requires one row axis",
            ));
        }
        window_elements(&out.source[..out.rank])?;
        out.elements = window_elements(&out.selected[..out.rank])?;
        Ok(out)
    }
    // The typed caller first authenticates the complete projection against its
    // original admission. This fixed adapter retains that actual local axis;
    // all ordinary row intersection and fragment arithmetic stays shared.
    pub(super) fn project_partition(mut self, projection: &CaptureSlicePartition, fragment: usize)
        -> Result<Self, CaptureWindowError>
    {
        let local = projection.fragments().get(fragment)
            .ok_or_else(|| invalid_fixed("partition window fragment is absent"))?.local();
        if self.partition.is_some() || projection.axis() >= self.rank
            || projection.global_shape() != &self.source[..self.rank]
            || projection.global_slice().shape != self.selected[..self.rank]
            || projection.local_shape().len() != self.rank
            || projection.local_shape()[self.axis] != self.source[self.axis]
            || local.shape.len() != self.rank || local.starts.len() != self.rank
            || local.ends.len() != self.rank || local.strides.len() != self.rank
        { return Err(invalid_fixed("partition window differs from its temporal source")); }
        for axis in 0..self.rank {
            if local.strides[axis] == 0 || local.starts[axis] > local.ends[axis]
                || local.ends[axis] > projection.local_shape()[axis]
                || local.shape[axis] != (local.ends[axis]-local.starts[axis]).div_ceil(local.strides[axis])
                || (axis != projection.axis() && (projection.local_shape()[axis] != self.source[axis]
                    || local.shape[axis] != self.selected[axis]))
            { return Err(invalid_fixed("partition window local geometry differs")); }
        }
        self.source[..self.rank].copy_from_slice(projection.local_shape());
        self.selected[..self.rank].copy_from_slice(&local.shape);
        self.start=local.starts[self.axis];self.end=local.ends[self.axis];self.stride=local.strides[self.axis];
        self.elements=window_elements(&self.selected[..self.rank])?;
        window_elements(&self.source[..self.rank])?;
        let axis=projection.axis();self.partition=Some((axis,local.starts[axis],local.strides[axis]));
        Ok(self)
    }
    /// Count globally anchored selected rows inside a physical span.
    pub fn local_rows(&self, start: u64, end: u64) -> Result<u64, CaptureError> {
        self.local_rows_fixed(start, end).map_err(Into::into)
    }
    /// Count through the same window worker without allocating a diagnostic.
    pub fn local_rows_fixed(&self, start: u64, end: u64) -> Result<u64, CaptureWindowError> {
        if start > end || end > self.source[self.axis] {
            return Err(invalid_fixed("capture window exceeds declared row extent"));
        }
        let begin = start.max(self.start);
        let limit = end.min(self.end);
        if begin >= limit {
            return Ok(0);
        }
        let first = window_add(
            self.start,
            window_mul((begin - self.start).div_ceil(self.stride), self.stride)?,
        )?;
        Ok(if first >= limit {
            0
        } else {
            (limit - first).div_ceil(self.stride)
        })
    }
    /// Validate an existing physical host record against this window geometry.
    pub fn validate_record(
        &self,
        record: &CaptureRecord,
        start: u64,
        end: u64,
    ) -> Result<u64, CaptureError> {
        self.validate_record_fixed(record, start, end)
            .map_err(Into::into)
    }
    /// Validate the same physical record with an allocation-free error value.
    pub fn validate_record_fixed(
        &self,
        record: &CaptureRecord,
        start: u64,
        end: u64,
    ) -> Result<u64, CaptureWindowError> {
        let rows = self.local_rows_fixed(start, end)?;
        let source = record
            .source_shape
            .as_deref()
            .ok_or_else(|| invalid_fixed("logical reduction hook was not emitted"))?;
        if source.len() != self.rank {
            return Err(invalid_fixed("reduction source rank changed"));
        }
        for (i, n) in source.iter().enumerate() {
            let expected = if i == self.axis {
                end - start
            } else {
                self.source[i]
            };
            if *n != expected {
                return Err(invalid_fixed("reduction source geometry changed"));
            }
        }
        let mut expected = self.selected;
        expected[self.axis] = rows;
        let count = window_elements(&expected[..self.rank])?;
        if (count != 0 || record.selected_shape.is_some())
            && record.selected_shape.as_deref() != Some(&expected[..self.rank])
        {
            return Err(invalid_fixed("reduction fragment selection changed"));
        }
        Ok(count)
    }
}

/// Fixed source-relative selection, shared by canonical prompt chunks and
/// independently attributed invocation windows. Geometry grants no authority.
#[derive(Debug)]
pub(super) struct WindowSelection {
    pub(super) source: [u64; 32],
    pub(super) selected: [u64; 32],
    pub(super) starts: [u64; 32],
    pub(super) ends: [u64; 32],
    pub(super) strides: [u64; 32],
    pub(super) row_offset: u64,
}
impl CaptureWindowGeometry {
    pub(super) fn fragment(
        &self,
        point: &crate::ObservationPoint,
        selection: &CaptureSelection,
        start: u64,
        end: u64,
    ) -> Result<WindowSelection, CaptureError> {
        let axes = point
            .axes
            .as_ref()
            .ok_or_else(|| invalid("window has no axes"))?;
        if axes.len() != self.rank {
            return Err(invalid("window axes changed"));
        }
        let mut source = self.source;
        let mut selected = self.selected;
        source[self.axis] = end.checked_sub(start).ok_or(CaptureError::Overflow)?;
        selected[self.axis] = self.local_rows(start, end)?;
        let lo = start
            .saturating_sub(self.start)
            .div_ceil(self.stride)
            .min(self.selected[self.axis]);
        let mut starts = [0; 32];
        let mut ends = [0; 32];
        let mut strides = [1; 32];
        for (i, declared) in axes.iter().enumerate() {
            if let Some(slice) = selection.slices.iter().find(|s| s.axis == declared.name) {
                starts[i] = slice.start;
                strides[i] = slice.stride;
            }
            if let Some((axis, start, stride)) = self.partition {
                if i == axis { starts[i] = start; strides[i] = stride; }
            }
            if i == self.axis && selected[i] != 0 {
                starts[i] = add(self.start, mul(lo, self.stride)?)?
                    .checked_sub(start)
                    .ok_or(CaptureError::Overflow)?;
            }
            if selected[i] == 0 {
                starts[i] = 0;
            }
            ends[i] = if selected[i] == 0 {
                0
            } else {
                add(add(starts[i], mul(selected[i] - 1, strides[i])?)?, 1)?
            };
            if ends[i] > source[i] {
                return Err(invalid("window selection exceeds source"));
            }
        }
        elements(&source[..self.rank])?;
        elements(&selected[..self.rank])?;
        Ok(WindowSelection {
            source,
            selected,
            starts,
            ends,
            strides,
            row_offset: lo,
        })
    }
}

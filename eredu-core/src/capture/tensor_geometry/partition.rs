//! Original global selection projected onto an actual native fragment.
use super::*;
impl<'a> CaptureTensorGeometry<'a> {
    /// Build the ordinary raw tensor geometry for an exact receipt fragment.
    /// Disjoint Summary/Histogram use their typed reduction geometry instead.
    /// Additive reductions deliberately export raw terms under the *original*
    /// admission; the shared assembly applies the nonlinear transform afterward.
    /// This descriptive projection grants no claim, source or native work.
    pub fn prepare_partition(
        source: &'a AdmittedCapturePlan, selection: usize, phase: CapturePhase,
        prediction: u64, invocation: Option<CaptureInvocationShape>,
        projection: &CaptureSlicePartition, fragment: usize,
        combination: PartitionCaptureCombination,
    ) -> Result<Self, CaptureTensorGeometryError> {
        let kind = match combination {
            PartitionCaptureCombination::Disjoint => GeometryTransform::Tensor,
            PartitionCaptureCombination::SumF64ToF32 => GeometryTransform::AdditiveTensor,
        };
        Self::prepare_kind(source, selection, phase, prediction, invocation, kind)?
            .apply_partition(projection, fragment, combination)
    }
    /// Compose an actual spatial fragment with a physical window of the same
    /// explicitly admitted logical invocation. Neither coordinate replaces the
    /// other: projection coordinates remain logical, native axes are physical.
    pub fn prepare_partition_window(
        source: &'a AdmittedCapturePlan, selection: usize, phase: CapturePhase,
        prediction: u64, physical: CaptureInvocationShape, window: CaptureInvocationWindow,
        projection: &CaptureSlicePartition, fragment: usize,
        combination: PartitionCaptureCombination,
    ) -> Result<Self, CaptureTensorGeometryError> {
        let kind = match combination {
            PartitionCaptureCombination::Disjoint => GeometryTransform::Tensor,
            PartitionCaptureCombination::SumF64ToF32 => GeometryTransform::AdditiveTensor,
        };
        Self::prepare_partition_window_kind(source, selection, phase, prediction,
            physical, window, projection, fragment, combination, kind)
    }
    pub(super) fn prepare_partition_window_kind(
        source: &'a AdmittedCapturePlan, selection: usize, phase: CapturePhase,
        prediction: u64, physical: CaptureInvocationShape, window: CaptureInvocationWindow,
        projection: &CaptureSlicePartition, fragment: usize,
        combination: PartitionCaptureCombination, kind: GeometryTransform,
    ) -> Result<Self, CaptureTensorGeometryError> {
        let logical = window.validate(physical)?;
        source.geometry_at(phase, prediction, Some(physical))?;
        let selected = Self::prepare_kind(source, selection, phase, prediction, Some(logical), kind)?;
        let rows = CaptureWindowGeometry::from_resolved(&source.points()[selection],
            &source.plan().selections[selection], physical.batch, logical.sequence, selected.source_shape())?;
        // Authenticate the complete spatial projection before letting the row
        // worker consume it. The fixed adapter rejects a changed local row
        // extent, so an independent row shard cannot impersonate this window.
        let selected = selected.apply_partition(projection, fragment, combination)?;
        let rows = rows.project_partition(projection, fragment).map_err(CaptureError::from)?;
        selected.apply_selected_window(physical, window, rows)
    }
    /// Exact additional coordinate/source frames for the composed window path.
    pub fn partition_window_preparation_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [Self::partition_preparation_control_bytes()?,
            Self::preparation_control_bytes()?, size_of::<Self>() * 2,
            size_of::<CaptureWindowGeometry>() * 2,
            size_of::<Result<CaptureWindowGeometry, CaptureWindowError>>(),
            size_of::<Result<CaptureWindowGeometry, CaptureError>>(),
            size_of::<(CaptureInvocationShape, CaptureInvocationShape, CaptureInvocationWindow)>(),
            size_of::<(&AdmittedCapturePlan, usize, CapturePhase, u64, CaptureInvocationShape,
                CaptureInvocationWindow, &CaptureSlicePartition, usize, PartitionCaptureCombination, GeometryTransform)>(),
        ];
        frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Fixed caller/projection frames in addition to ordinary geometry controls.
    /// Callers reserve these before preparing a fragment or reduction source.
    pub fn partition_preparation_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [Self::preparation_control_bytes()?, size_of::<Self>() * 2,
            size_of::<Result<Self, CaptureTensorGeometryError>>(),
            size_of::<(&AdmittedCapturePlan, usize, CapturePhase, u64, Option<CaptureInvocationShape>,
                &CaptureSlicePartition, usize, PartitionCaptureCombination)>(),
            size_of::<(&ResolvedCaptureSlice, &CaptureFragmentGeometry, &ResolvedCaptureSlice)>(),
            size_of::<(usize, usize, u64, bool, GeometryTransform)>(),
            size_of::<std::ops::Range<usize>>(),
        ];
        frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(super) fn apply_partition(mut self, projection: &CaptureSlicePartition,
        fragment: usize, combination: PartitionCaptureCombination)
        -> Result<Self, CaptureTensorGeometryError>
    {
        use CaptureTensorGeometryError as E;
        let rank = self.source_rank;
        let global = projection.global_slice();
        if projection.global_shape().len() != rank || projection.local_shape().len() != rank
            || global.starts != self.starts() || global.ends != self.ends() || global.strides != self.strides()
            || global.shape.len() != rank
            || !projection.global_shape().iter().zip(self.source_shape()).all(|(a,b)| usize::try_from(*a).ok() == Some(*b))
        { return Err(E::Partition); }
        for axis in 0..rank {
            if global.shape[axis] != (self.ends[axis]-self.starts[axis]).div_ceil(self.strides[axis]) {
                return Err(E::Partition);
            }
        }
        let selected = projection.fragments().get(fragment).ok_or(E::Partition)?;
        let local = selected.local();
        if combination == PartitionCaptureCombination::SumF64ToF32 {
            let destination = selected.destination();
            if projection.fragments().len() != 1 || projection.local_shape() != projection.global_shape()
                // The contiguous worker normalizes its exclusive end to the
                // last selected coordinate + 1. Equal starts/strides/counts
                // prove the same ordered coordinates even when that end is
                // below the original admitted slice end (e.g. 1..7 step 2).
                // Match the ordinary additive receipt's coordinate contract.
                || local.starts != global.starts || local.strides != global.strides || local.shape != global.shape
                || destination.shape != global.shape || destination.ends != global.shape
                || destination.starts.iter().any(|n| *n != 0) || destination.strides.iter().any(|n| *n != 1)
            { return Err(E::Partition); }
        }
        self.elements = 1;
        self.rank = rank;
        for axis in 0..rank {
            self.source_shape[axis] = usize::try_from(projection.local_shape()[axis]).map_err(|_| E::Overflow)?;
            self.starts[axis] = local.starts[axis]; self.ends[axis] = local.ends[axis]; self.strides[axis] = local.strides[axis];
            self.shape[axis] = usize::try_from(local.shape[axis]).map_err(|_| E::Overflow)?;
            self.elements = self.elements.checked_mul(self.shape[axis]).ok_or(E::Overflow)?;
        }
        if let CaptureTransform::Preview { max_elements } = self.native_transform() {
            self.elements = self.elements.min(usize::try_from(*max_elements).map_err(|_| E::Overflow)?);
            self.shape = [0; 32]; self.shape[0] = self.elements; self.rank = 1;
        }
        Ok(self)
    }
}

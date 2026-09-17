//! Exact row and reached-context projection; destinations belong to the caller.
use super::*;
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum InterventionPrefillProjectionError {
    #[error(transparent)]
    Source(#[from] InterventionPrefillSourceError),
    #[error(transparent)]
    Window(#[from] CaptureWindowSourceError),
    #[error(transparent)]
    Axes(#[from] CaptureAxisError),
    #[error(transparent)]
    Geometry(#[from] InterventionGeometryError),
    #[error(transparent)]
    Projection(#[from] CaptureContiguousProjectionError),
}
impl InterventionPrefillWindow {
    /// Resolve the immutable ordinary selection, then intersect its row range
    /// and Context axes with this exact reached source. The payload destination
    /// remains relative to the original selected tensor, never the current prefix.
    pub fn resolve_projection_into(
        self,
        plan: &AdmittedInterventionPlan,
        index: usize,
        actual: &[u64],
        dtype: InterventionDtype,
        global: &mut [u64],
        selected: &mut ResolvedCaptureSlice,
        local: &mut ResolvedCaptureSlice,
        destination: &mut ResolvedCaptureSlice,
    ) -> Result<(usize, bool), InterventionPrefillProjectionError> {
        self.validate(plan)?;
        let point = plan
            .points()
            .get(index)
            .ok_or(InterventionPrefillSourceError::Identity)?;
        if point.axes.len() > 32 {
            return Err(CaptureAxisError::RankBound.into());
        }
        let physical = self.physical();
        let (_, row) =
            self.window()
                .source_axes_into(physical, Some(&point.axes), actual, global)?;
        let logical = plan
            .geometry_at(CapturePhase::Prefill, 0, None)
            .map_err(|_| InterventionPrefillSourceError::Identity)?;
        let mut ranges: [(usize, Range<u64>); 32] = std::array::from_fn(|_| (0, 0..0));
        ranges[0] = (row, self.range[0]..self.range[1]);
        let mut count = 1;
        for (axis, declaration) in point.axes.iter().enumerate() {
            if declaration.dimension == SymbolicDimension::Context {
                global[axis] = logical
                    .context
                    .ok_or(InterventionPrefillSourceError::Profile)?;
                ranges[count] = (
                    axis,
                    0..physical
                        .context
                        .ok_or(InterventionPrefillSourceError::Profile)?,
                );
                count += 1;
            }
        }
        logical.validate_actual_axes(Some(&point.axes), global)?;
        plan.resolve_prepared_at(index, CapturePhase::Prefill, 0, global, dtype, selected)?;
        let overlap = CaptureSlicePartition::contiguous_axes_into(
            global,
            selected,
            &ranges[..count],
            local,
            destination,
        )?;
        Ok((row, overlap))
    }
    /// Fixed shared resolver controls; exact-rank Vec backing and any projected
    /// payload are reserved separately by their existing owners before use.
    pub fn projection_control_bytes() -> Option<usize> {
        let frames = [
            Self::control_bytes()?,
            CaptureInvocationWindow::source_axes_control_bytes()?,
            CaptureSlicePartition::contiguous_projection_control_bytes()?,
            size_of::<[(usize, Range<u64>); 32]>(),
            size_of::<[CaptureInvocationShape; 2]>(),
            size_of::<Result<CaptureInvocationShape, CaptureError>>(),
            size_of::<(
                &AdmittedInterventionPlan,
                CapturePhase,
                u64,
                Option<CaptureInvocationShape>,
            )>(),
            size_of::<(usize, usize)>(),
            size_of::<Result<(usize, bool), InterventionPrefillProjectionError>>(),
            size_of::<InterventionPrefillProjectionError>(),
            size_of::<(
                Self,
                &AdmittedInterventionPlan,
                usize,
                &[u64],
                InterventionDtype,
                &mut [u64],
                &mut ResolvedCaptureSlice,
                &mut ResolvedCaptureSlice,
                &mut ResolvedCaptureSlice,
            )>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}

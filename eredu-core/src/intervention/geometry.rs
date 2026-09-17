//! Fixed destination geometry for admitted activation programs.
use super::*;

/// Allocation-free preparation refusal; it conveys no execution authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum InterventionGeometryError {
    /// The coordinate, admission mode, or exact dtype differs.
    #[error("intervention coordinate or exact dtype differs from admission")]
    Coordinate,
    /// The actual source does not match the declared axes.
    #[error("intervention source axis mismatch: {0}")]
    Axes(#[from] CaptureAxisError),
    /// The supplied slice/destination does not match the source.
    #[error("intervention slice mismatch: {0}")]
    Slice(#[from] CaptureSliceDestinationError),
    /// The source selection differs from the exact payload shape.
    #[error("intervention payload shape differs from selected source")]
    Payload,
}
impl AdmittedInterventionPlan {
    /// Resolve this ordinary-geometry admission into four preallocated rank
    /// destinations. Exact symbolic axes, half-open slicing and payload matching
    /// use the same workers as `validate_at`; invocation imports remain distinct.
    pub fn resolve_prepared_at(
        &self,
        index: usize,
        phase: CapturePhase,
        prediction: u64,
        shape: &[u64],
        dtype: InterventionDtype,
        destination: &mut ResolvedCaptureSlice,
    ) -> Result<(), InterventionGeometryError> {
        self.resolve_prepared_geometry(index, phase, prediction, None, shape, dtype, destination)
    }
    /// Resolve the same shared slice/payload worker for one admitted physical
    /// invocation. The caller must separately authenticate its source and role.
    pub fn resolve_prepared_invocation_at(
        &self,
        index: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: CaptureInvocationShape,
        shape: &[u64],
        dtype: InterventionDtype,
        destination: &mut ResolvedCaptureSlice,
    ) -> Result<(), InterventionGeometryError> {
        self.resolve_prepared_geometry(
            index,
            phase,
            prediction,
            Some(invocation),
            shape,
            dtype,
            destination,
        )
    }
    fn resolve_prepared_geometry(
        &self,
        index: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        shape: &[u64],
        dtype: InterventionDtype,
        destination: &mut ResolvedCaptureSlice,
    ) -> Result<(), InterventionGeometryError> {
        use InterventionGeometryError as E;
        let operation = self.plan.operations.get(index).ok_or(E::Coordinate)?;
        let point = self.points.get(index).ok_or(E::Coordinate)?;
        if prediction >= self.request.max_predictions
            || !operation.schedule.includes(phase, prediction)
            || operation.action.dtype() != Some(dtype)
        {
            return Err(E::Coordinate);
        }
        let actual = match (self.invocation_bounds, invocation) {
            (None, None) => self
                .text_origin
                .invocation_shape(self.request, phase, prediction)
                .map_err(|_| E::Coordinate)?,
            (Some(bounds), Some(actual)) => {
                bounds.validate_fixed(actual, prediction)?;
                actual
            }
            _ => return Err(E::Coordinate),
        };
        actual.validate_actual_axes(Some(&point.axes), shape)?;
        resolve_slice_into(
            Some(&point.axes),
            &operation.slices,
            shape,
            &mut destination.starts,
            &mut destination.ends,
            &mut destination.strides,
            &mut destination.shape,
        )?;
        if !payload_shape_matches(&operation.action, &destination.shape) {
            return Err(E::Payload);
        }
        Ok(())
    }
}

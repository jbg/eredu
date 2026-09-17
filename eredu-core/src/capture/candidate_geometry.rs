//! Borrowed terminal-row geometry; no source pin, execution or allocation grant.
use super::*;
use crate::{ObservationDtype, ObservationValueType, SymbolicDimension};

/// Shared raw model-logit terminal geometry. Selection-specific output facts
/// are validated by the typed caller; this supplies no native source custody.
pub(super) fn terminal_shape(
    source: &AdmittedCapturePlan,
    index: usize,
    phase: CapturePhase,
    prediction: u64,
    invocation: Option<CaptureInvocationShape>,
) -> Result<[usize; 3], CaptureTensorGeometryError> {
    let selection = source
        .plan()
        .selections
        .get(index)
        .ok_or(CaptureTensorGeometryError::SelectionMissing { index })?;
    let point = &source.points()[index];
    if prediction >= source.request().max_predictions
        || !selection.schedule.includes(phase, prediction)
        || !match phase {
            CapturePhase::Prefill => point.prefill,
            CapturePhase::Decode => point.decode,
        }
    {
        return Err(CaptureTensorGeometryError::Inactive);
    }
    let axes = point
        .axes
        .as_ref()
        .ok_or(CaptureTensorGeometryError::UnknownShape)?;
    if selection.path != crate::MODEL_LOGITS_OBSERVATION_PATH
        || !selection.slices.is_empty()
        || point.dtype != ObservationDtype::Floating
        || point.value_type != ObservationValueType::Tensor
        || axes.len() != 3
        || axes[0].dimension != SymbolicDimension::Batch
        || axes[1].dimension != SymbolicDimension::Sequence
    {
        return Err(CaptureTensorGeometryError::Unsupported);
    }
    let logical = source.geometry_at(phase, prediction, invocation)?;
    logical.validate()?;
    let mut shape = [0; 3];
    for (out, axis) in shape.iter_mut().zip(axes) {
        *out = usize::try_from(
            logical
                .extent(&axis.dimension)?
                .ok_or(CaptureTensorGeometryError::UnknownShape)?,
        )
        .map_err(|_| CaptureTensorGeometryError::Overflow)?;
    }
    if shape[0] != 1 || shape[1] == 0 || shape[2] == 0 || shape[2] > i32::MAX as usize {
        return Err(CaptureTensorGeometryError::Unsupported);
    }
    Ok(shape)
}

/// Exact existing TopCandidates source and fixed host destination geometry.
/// The logical invocation may contain many rows; only its terminal row is read.
#[derive(Debug)]
pub struct CaptureCandidateGeometry<'a> {
    source: &'a AdmittedCapturePlan,
    index: usize,
    phase: CapturePhase,
    prediction: u64,
    shape: [usize; 3],
    count: usize,
}
impl<'a> CaptureCandidateGeometry<'a> {
    /// Validate an actual immutable selection without allocating or cloning it.
    pub fn prepare(
        source: &'a AdmittedCapturePlan,
        index: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
    ) -> Result<Self, CaptureTensorGeometryError> {
        let shape = terminal_shape(source, index, phase, prediction, invocation)?;
        let CaptureTransform::TopCandidates { count } = source.plan().selections[index].transform
        else {
            return Err(CaptureTensorGeometryError::Unsupported);
        };
        let count = usize::try_from(count).map_err(|_| CaptureTensorGeometryError::Overflow)?;
        if count == 0 || count > shape[2] {
            return Err(CaptureTensorGeometryError::Unsupported);
        }
        Ok(Self {
            source,
            index,
            phase,
            prediction,
            shape,
            count,
        })
    }
    /// Select the actual terminal physical readout after its canonical span is
    /// authenticated by the caller. This pure geometry supplies no such authority.
    pub fn terminal_readout(mut self, rows: usize) -> Result<Self, CaptureTensorGeometryError> {
        if rows == 0 || rows > self.shape[1] {
            return Err(CaptureTensorGeometryError::Unsupported);
        }
        self.shape[1] = rows;
        Ok(self)
    }
    /// Original immutable source, never a native source pin.
    pub fn admission(&self) -> &'a AdmittedCapturePlan {
        self.source
    }
    /// Original selection ordinal.
    pub fn selection_index(&self) -> usize {
        self.index
    }
    /// Original phase.
    pub fn phase(&self) -> CapturePhase {
        self.phase
    }
    /// Original run-relative prediction, unchanged by chunking.
    pub fn prediction(&self) -> u64 {
        self.prediction
    }
    /// Actual expected physical source shape, including selected readout rows.
    pub fn source_shape(&self) -> &[usize; 3] {
        &self.shape
    }
    /// Complete vocabulary width, including IDs outside a sampler domain.
    pub fn vocabulary(&self) -> usize {
        self.shape[2]
    }
    /// Exact number of descending candidate slots.
    pub fn count(&self) -> usize {
        self.count
    }
}

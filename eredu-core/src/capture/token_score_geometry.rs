//! Borrowed selected-token scores on the existing terminal model-logit row.
use super::*;

/// Exact ordered ID selection and actual terminal raw-logit source geometry.
/// The complete vocabulary remains the normalization and ranking domain.
#[derive(Debug)]
pub struct CaptureTokenScoreGeometry<'a> {
    source: &'a AdmittedCapturePlan,
    index: usize,
    phase: CapturePhase,
    prediction: u64,
    shape: [usize; 3],
    token_ids: &'a [u32],
}
impl<'a> CaptureTokenScoreGeometry<'a> {
    /// Resolve the actual admitted selection without copying its ID buffer.
    pub fn prepare(
        source: &'a AdmittedCapturePlan,
        index: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
    ) -> Result<Self, CaptureTensorGeometryError> {
        let shape =
            candidate_geometry::terminal_shape(source, index, phase, prediction, invocation)?;
        let CaptureTransform::TokenScores { token_ids } =
            &source.plan().selections[index].transform
        else {
            return Err(CaptureTensorGeometryError::Unsupported);
        };
        if token_ids.is_empty() || token_ids.iter().any(|&id| id as usize >= shape[2]) {
            return Err(CaptureTensorGeometryError::Unsupported);
        }
        Ok(Self {
            source,
            index,
            phase,
            prediction,
            shape,
            token_ids,
        })
    }
    /// Actual terminal physical readout after separate canonical-span validation.
    pub fn terminal_readout(mut self, rows: usize) -> Result<Self, CaptureTensorGeometryError> {
        if rows == 0 || rows > self.shape[1] {
            return Err(CaptureTensorGeometryError::Unsupported);
        }
        self.shape[1] = rows;
        Ok(self)
    }
    /// Original immutable semantic source, never a native pin or account.
    pub fn admission(&self) -> &'a AdmittedCapturePlan {
        self.source
    }
    /// Original selection ordinal.
    pub fn selection_index(&self) -> usize {
        self.index
    }
    /// Original logical phase.
    pub fn phase(&self) -> CapturePhase {
        self.phase
    }
    /// Run-relative prediction unchanged by physical chunking.
    pub fn prediction(&self) -> u64 {
        self.prediction
    }
    /// Actual physical source shape.
    pub fn source_shape(&self) -> &[usize; 3] {
        &self.shape
    }
    /// Complete model vocabulary including IDs outside a decision domain.
    pub fn vocabulary(&self) -> usize {
        self.shape[2]
    }
    /// Ordered IDs borrowed from this exact admitted selection.
    pub fn token_ids(&self) -> &'a [u32] {
        self.token_ids
    }
    /// Fixed number of result slots.
    pub fn count(&self) -> usize {
        self.token_ids.len()
    }
}

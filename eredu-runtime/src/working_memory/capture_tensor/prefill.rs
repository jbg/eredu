//! Private one-buffer initialized scatter storage. Logical coverage is separate.
use super::*;
use eredu_core::{InferenceGeometry, capture::*, checkpoint::TensorDtype};

/// Typed rejection by the closed host fragment program, not native authority.
#[derive(Debug, thiserror::Error)]
pub enum CapturePrefillHostError {
    /// Shared logical hook progression, independent of destination coverage.
    #[error(transparent)]
    Progression(#[from] crate::capture::CapturePrefillProgressError),
    /// The actual immutable source does not admit this row mapping.
    #[error(transparent)]
    Geometry(#[from] CapturePrefillGeometryError),
    /// The exact additive transform or projected window source was rejected.
    #[error(transparent)]
    Partition(#[from] CapturePrefillPartitionError),
    /// Another source, selection, phase or inference schedule was supplied.
    #[error("prefill capture source, schedule or selection differs")]
    Identity,
    /// Inactive, spent, failed or concurrently borrowed target.
    #[error("prefill capture target {index} is not available")]
    Target {
        /// Original selection index.
        index: usize,
    },
    /// Fragment order cannot refill or duplicate a destination contribution.
    #[error("prefill capture fragment is duplicate or out of order")]
    Order,
    /// Zero-initialized destination storage is not complete logical coverage.
    #[error("prefill capture target {index} has incomplete coverage")]
    Incomplete {
        /// Original selection index.
        index: usize,
    },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::working_memory) enum TargetState {
    Inactive,
    Unstarted,
    Claimed,
    Active,
    // Original full target is assigned to the retained remote producer. Chunk
    // progress is real; no local scalar or zero-filled coverage is asserted.
    Remote,
    // The once-only final receipt claim has been issued. Drop/error cannot retry.
    RemoteClaimed,
    // A source-checked host receipt exists; final frame sealing remains separate.
    RemoteRecorded,
    /// Local terms live in their paid partition bank until global assembly.
    Assembling,
    /// The global assembly consumed its one final host claim.
    AssemblyClaimed,
    /// Global assembly and its delivery vote completed before frame sealing.
    AssemblyRecorded,
    Failed,
    Recorded,
}
#[derive(Debug)]
pub(in crate::working_memory) struct TargetSlot {
    pub tensor: Option<OwnedPrefillTensor>,
    pub completed: Option<SharedTensorObservation>,
    pub summary: Option<crate::capture::reduction::Summary>,
    pub histogram: Option<eredu_core::capture::CaptureHistogram>,
    pub routed: Option<crate::working_memory::capture_run::OwnedCaptureRoutedUnits>,
    pub routed_covered: u64,
    pub state: TargetState,
    pub dtype: Option<TensorDtype>,
    pub done: bool,
    pub progression: Option<crate::capture::CapturePrefillRowProgress>,
}
#[derive(Debug)]
pub(in crate::working_memory) struct PrefillTargets {
    pub slots: Box<[TargetSlot]>,
    pub inference: InferenceGeometry,
    pub next: u64,
}
impl PrefillTargets {
    pub fn complete(&self) -> bool {
        self.next
            == self
                .inference
                .input_positions
                .div_ceil(self.inference.prefill_chunk_positions)
            && self
                .slots
                .iter()
                .all(|s| matches!(s.state, TargetState::Inactive | TargetState::Recorded))
    }
}
#[derive(Debug)]
pub(in crate::working_memory) struct OwnedPrefillTensor {
    // Actual payload always retires before the final original-H custody.
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub covered: usize,
    pub custody: CaptureTensorCustody,
}
impl OwnedPrefillTensor {
    pub fn allocate(
        plan: CaptureTensorHostPlan<'_>,
        custody: CaptureTensorCustody,
    ) -> Result<Self, WorkingMemoryError> {
        Self::allocate_geometry(&plan.geometry, custody)
    }
    // Both callers hold the exact whole-value host plan before allocation. The
    // partition adapter supplies its authenticated local geometry, while the
    // ordinary entry above retains the original global geometry.
    pub(in crate::working_memory) fn allocate_geometry(
        geometry: &CaptureTensorGeometry<'_>, custody: CaptureTensorCustody,
    ) -> Result<Self, WorkingMemoryError> {
        custody.validate()?;
        let mut shape = Vec::with_capacity(geometry.shape().len());
        for &n in geometry.shape() {
            shape.push(n);
        }
        let mut data = Vec::with_capacity(geometry.elements());
        // One exact initialized destination; these placeholders prove no coverage.
        data.resize(geometry.elements(), 0.0);
        custody.validate()?;
        Ok(Self {
            shape,
            data,
            covered: 0,
            custody,
        })
    }
}

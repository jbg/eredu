//! Runtime-issued coordinates for a capture-only source segment.
use crate::{prefill::PrefillChunk, working_memory::InferenceRequest};
use eredu_core::{DistributedCommitEpoch, InferenceGeometry};

pub use crate::working_memory::{PreparedPrefillChunkRetention, SettledPrefillChunkRetention};

/// Borrowed exact upcoming chunk of the existing session transaction. Only the
/// shared runtime scheduler constructs this context. It neither advances the
/// epoch nor grants allocation, capture claims, or completion authority.
#[derive(Debug)]
pub struct PrefillChunkRetentionContext<'a> {
    request: &'a InferenceRequest,
    chunk: &'a PrefillChunk,
    epoch: DistributedCommitEpoch,
}
impl<'a> PrefillChunkRetentionContext<'a> {
    pub(crate) fn new(
        request: &'a InferenceRequest,
        chunk: &'a PrefillChunk,
        epoch: DistributedCommitEpoch,
    ) -> Self {
        Self {
            request,
            chunk,
            epoch,
        }
    }
    /// Exact original request; not an independently issued reservation.
    pub fn request(&self) -> &'a InferenceRequest {
        self.request
    }
    /// Actual fixed-schedule range, absolute position and demanded readout.
    pub fn chunk(&self) -> &'a PrefillChunk {
        self.chunk
    }
    /// Checked upcoming epoch. The real session increment remains unchanged.
    pub fn epoch(&self) -> DistributedCommitEpoch {
        self.epoch
    }
    /// Original request geometry, including cached origin and chunk width.
    pub fn geometry(&self) -> InferenceGeometry {
        self.request.geometry()
    }
}

mod opening_state;
pub use opening_state::PrefillOpeningState;
pub(crate) use opening_state::RuntimeOpeningState;

mod opening_execution;
pub use opening_execution::PrefillOpeningExecution;
pub(crate) use opening_execution::RuntimeOpeningExecution;

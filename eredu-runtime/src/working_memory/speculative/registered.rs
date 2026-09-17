//! Registered input provenance is distinct from model and numerical births.
use super::*;
use crate::working_memory::WorkspaceCopyRetention;

/// An exact request association for an already admitted copy account.
/// This is descriptive provenance only: no native allocation, completion,
/// registered root, model role, or numerical phase is authorized by this value.
/// A backend's closed published-copy owner authenticates its actual storage.
#[derive(Clone, Debug)]
pub struct OriginalSpeculativeRegisteredSource {
    identity: OriginalSpeculativeSourceIdentity,
    // Copy account and its source/preparation custody outlive covered resources.
    _copy: WorkspaceCopyRetention,
}
impl OriginalSpeculativeRegisteredSource {
    pub(super) fn identity(&self) -> &OriginalSpeculativeSourceIdentity {
        &self.identity
    }
}
impl OriginalSpeculativeRequest {
    /// Associate an actual copy retention with this original request.
    /// The existing copy account remains charged independently; it is never
    /// relabeled as a model/numerical birth or deducted from an equation quote.
    /// The caller must separately validate a closed completed native copy.
    pub fn bind_registered_copy_source(
        &self,
        copy: &WorkspaceCopyRetention,
    ) -> Result<OriginalSpeculativeRegisteredSource, WorkingMemoryError> {
        copy.validate_pool(self.ticket.pool())?;
        let slots = self
            .slots
            .try_lock()
            .map_err(|_| WorkingMemoryError::AccountConstructionBusy)?;
        if slots.closed {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        drop(slots);
        Ok(OriginalSpeculativeRegisteredSource {
            identity: self.source_identity(),
            _copy: copy.clone(),
        })
    }
}

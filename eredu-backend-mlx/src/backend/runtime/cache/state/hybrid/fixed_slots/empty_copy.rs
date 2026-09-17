//! Fresh zero-payload child tables under the same admitted host constructor.
use super::{FixedStateSlots, Slot};
use crate::backend::error::Error;
use eredu_core::{BackendFailure, HostPreparationAuthority};
use eredu_runtime::{DenseHostSlotInitialization, PreparedDenseHostCopyError};
use eredu_runtime::working_memory::WorkingMemoryError;
use std::mem::size_of;

// A validated empty source never calls this callback. Keep a typed refusal if
// that invariant changes; no source value or role can disappear in this worker.
fn empty_value(_: usize, _: &Slot) -> Result<Slot, WorkingMemoryError> {
    Err(WorkingMemoryError::IdentityMismatch)
}

impl FixedStateSlots {
    fn empty_source(&self) -> Option<DenseHostSlotInitialization<'_, Slot>> {
        if !self.is_empty() || self.payload_bytes() != Some(0) {
            return None;
        }
        self.prepare_slots().ok()?.for_dense_destination().ok()
    }

    /// Actual source-selected shared constructor, typed error custody and
    /// transport frames. There is no child slot or numerical payload.
    pub(in crate::backend::runtime::cache::state::hybrid) fn empty_copy_preparation_bytes(&self) -> Option<usize> {
        let source = self.empty_source()?;
        [
            usize::try_from(source.prepared_copy_bytes::<WorkingMemoryError, _>(&empty_value)?).ok()?,
            BackendFailure::source_retention_peak_bytes::<
                PreparedDenseHostCopyError<WorkingMemoryError>,
            >()?,
            size_of::<Self>(),
            size_of::<Option<DenseHostSlotInitialization<'_, Slot>>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Option<&HostPreparationAuthority>>(),
            size_of::<eredu_runtime::HostSlotInitializationError>(),
            size_of::<Result<eredu_runtime::HostSlotInitialization<'_, Slot>,
                eredu_runtime::HostSlotInitializationError>>(),
            size_of::<Result<DenseHostSlotInitialization<'_, Slot>,
                eredu_runtime::HostSlotInitializationError>>(),
        ].into_iter().try_fold(size_of::<[usize; 9]>(), usize::checked_add)
    }

    /// The enclosing exact source query has paid this child's metadata. The
    /// shared worker mints a fresh identity retaining H through every alias.
    pub(in crate::backend::runtime::cache::state::hybrid) fn copy_empty_prepared(
        &self,
        host: &HostPreparationAuthority,
    ) -> Result<Self, Error> {
        let source = self.empty_source()
            .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
        let slots = source.copy_with_preparation(host, empty_value)
            .map_err(|cause| Error::StorageSource(BackendFailure::from_error(cause)))?;
        Ok(Self::from_published_slots(slots))
    }
}

#[cfg(test)]
mod tests;

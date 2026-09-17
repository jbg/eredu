//! Closed existing-only group installation; completeness remains a caller proof.
use super::*;
use crate::inspection::PrefillChunkRetentionContext;
use crate::working_memory::{BoundedPinError, BoundedRegisteredStorage};
use std::sync::TryLockError;

impl CaptureSourceSegment {
    /// Install one successful, originally priced group for this exact chunk.
    ///
    /// The caller must stage its owner in `pending` before invoking this method.
    /// Busy, rejection and provider-comparison unwind leave that owner untouched.
    /// Only success consumes it, by moving the existing Arc into the exact slot.
    /// Clones remain ordinary shared custody; a second installation is rejected.
    ///
    /// This publishes no new key and proves neither complete opening inventory
    /// nor native completion. All sources must already be registered. No scalar
    /// grant, hold, refund, certificate or raw registration is exposed.
    pub fn install_opening_group<K: Ord + Send + Sync + 'static>(
        &self,
        native: &mut WorkingMemoryFundingScope,
        context: &PrefillChunkRetentionContext<'_>,
        pending: &mut Option<BoundedRegisteredStorage<K>>,
    ) -> Result<(), BoundedPinError> {
        let group = pending
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let pool = native.pool.clone();
        let usage = pool.0.usage.try_lock().map_err(|error| match error {
            TryLockError::WouldBlock => BoundedPinError::Busy,
            TryLockError::Poisoned(_) => WorkingMemoryError::Poisoned.into(),
        })?;
        let slot = self.slot(native)?;
        if slot.opening.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        group.validate_install_locked(native, &usage, self, context)?;
        slot.validate_sources(&pool, &usage)?;
        // No callbacks, provider comparisons, allocations or fallible work after
        // this frontier. Both Options were checked while their owners were lent.
        let opening = pending
            .take()
            .expect("validated pending group")
            .into_opening();
        native
            .capture_source
            .as_mut()
            .expect("validated source slot")
            .opening = Some(opening);
        drop(usage);
        Ok(())
    }
}

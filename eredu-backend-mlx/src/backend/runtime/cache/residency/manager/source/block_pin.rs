//! Scalar canonical pin/lease owners for the exact prepared scan source.
use super::*;
use std::mem::size_of;

/// A source pin alone grants no tensor construction, promotion or execution.
/// It holds the actual manager's canonical backing and pool charge stable.
pub(crate) struct PinnedCacheBlock {
    id: CacheBlockId,
    manager: CacheResidencyManager,
    generation: u64,
    _funding: Option<HostMetadataFunding>,
}
/// One actual demand access to an already pinned canonical source. The owner
/// stays in the enclosing Q program until its numerical use has completed.
pub(crate) struct PinnedCacheBlockLease {
    id: CacheBlockId,
    manager: CacheResidencyManager,
    _funding: Option<HostMetadataFunding>,
}
impl CacheBlockSourceLoan<'_> {
    /// The scan's source program prepays fixed_controls before native entry.
    /// The loan validates the real selected catalog; no array or worker escapes.
    pub(crate) fn pin_prepared_block(
        &mut self,
        id: &CacheBlockId,
        funding: Option<HostMetadataFunding>,
    ) -> Result<PinnedCacheBlock, CacheSourceError> {
        if !self.blocks().any(|block| block.id() == id) {
            return Err(CacheSourceError::Identity);
        }
        self.lifecycle.pin_source(id)?;
        Ok(PinnedCacheBlock {
            id: id.clone(),
            manager: self.manager.clone(),
            generation: self.generation,
            _funding: funding,
        })
    }
}
impl PinnedCacheBlock {
    pub(crate) fn id(&self) -> &CacheBlockId {
        &self.id
    }
    /// Same canonical acquire worker/access clock as ordinary demand. Actual
    /// native completion/source checks precede this in the scan consumer.
    pub(crate) fn acquire(&self) -> Result<PinnedCacheBlockLease, CacheSourceError> {
        let started = Instant::now();
        let mut state = self
            .manager
            .inner
            .state
            .try_lock()
            .map_err(|cause| match cause {
                TryLockError::WouldBlock => CacheSourceError::Busy,
                TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
            })?;
        if state.generation != self.generation
            || !self.manager.borrowed_storage_complete(&state)
            || state
                .blocks
                .get(&self.id)
                .is_none_or(|record| record.physical.phase() != CacheStoragePhase::Device)
            || state.lifecycle.source_pin_count(&self.id)? == 0
        {
            return Err(CacheSourceError::Identity);
        }
        state
            .telemetry
            .validate_prepared_layers(std::iter::once(self.id.global_layer))
            .map_err(CacheLifecycleError::from)?;
        state.lifecycle.acquire(&self.id)?;
        // Same demand/access clock and telemetry; no promotion was needed.
        acquisition::record_device_hit(&mut state, &self.id, started.elapsed());
        drop(state);
        Ok(PinnedCacheBlockLease {
            id: self.id.clone(),
            manager: self.manager.clone(),
            _funding: self._funding.clone(),
        })
    }
    pub(crate) fn fixed_controls() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            CacheResidencyManager::original_discard_control_bytes()?,
            size_of::<PinnedCacheBlockLease>(),
            size_of::<CacheBlockId>(),
            size_of::<(
                &mut CacheBlockSourceLoan<'_>,
                &CacheBlockId,
                Option<HostMetadataFunding>,
            )>(),
            size_of::<Result<Self, CacheSourceError>>(),
            size_of::<Result<PinnedCacheBlockLease, CacheSourceError>>(),
            size_of::<MutexGuard<'_, CacheManagerState>>(),
            size_of::<
                Result<
                    MutexGuard<'_, CacheManagerState>,
                    TryLockError<MutexGuard<'_, CacheManagerState>>,
                >,
            >(),
            size_of::<Result<(), CacheLifecycleError>>(),
            size_of::<(u64, usize)>(),
            size_of::<std::iter::Once<&CacheBlockId>>(),
            size_of::<std::iter::Once<usize>>(),
            size_of::<Instant>(),
            size_of::<std::time::Duration>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}
impl Drop for PinnedCacheBlock {
    fn drop(&mut self) {
        pin::release_pins(&self.manager, std::iter::once(&self.id), true);
    }
}
impl Drop for PinnedCacheBlockLease {
    fn drop(&mut self) {
        pin::release_pins(&self.manager, std::iter::once(&self.id), false);
    }
}

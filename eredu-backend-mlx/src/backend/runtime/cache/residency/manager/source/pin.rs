//! Source pins reuse canonical lease ownership without demand telemetry.
use super::*;

/// An immutable source selection pinned in the actual manager. It keeps every
/// selected canonical backing and its finite cache-pool charge alive. It is
/// neither a completed native role nor permission to promote a block.
pub(crate) struct PinnedCacheSource {
    ids: Vec<CacheBlockId>,
    generation: u64,
    selection: CacheBlockSelection,
    manager: CacheResidencyManager,
    _context: WorkspaceContext,
}
impl PinnedCacheSource {
    pub(crate) fn selection(&self) -> CacheBlockSelection {
        self.selection
    }
    pub(crate) fn ids(&self) -> &[CacheBlockId] {
        &self.ids
    }
    pub(crate) fn manager(&self) -> &CacheResidencyManager {
        &self.manager
    }
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }
    /// Revalidates the actual catalog, not a descriptive byte-equivalent input.
    /// A generation change refuses reuse even if every selected ID still exists.
    pub(crate) fn validate_loan(
        &self,
        loan: &CacheBlockSourceLoan<'_>,
    ) -> Result<(), CacheSourceError> {
        if self.manager.session_id != loan.manager.session_id
            || !Arc::ptr_eq(&self.manager.inner, &loan.manager.inner)
            || self.generation != loan.generation
            || self.selection != loan.selection
            || !self.ids.iter().eq(loan.blocks().map(|block| block.id()))
        {
            return Err(CacheSourceError::Identity);
        }
        for id in &self.ids {
            if loan.lifecycle.source_pin_count(id)? == 0 {
                return Err(CacheSourceError::Identity);
            }
        }
        Ok(())
    }
    pub(crate) fn control_bytes(blocks: usize) -> Option<usize> {
        fixed_bytes()?.checked_add(WorkspaceContext::metadata_vec_bytes::<CacheBlockId>(
            blocks,
        )?)
    }
}
impl CacheBlockSourceLoan<'_> {
    /// Allocates the final ID inventory before acquiring any canonical pin.
    /// The returned owner must be retained outside this manager loan, including
    /// on error/unwind: dropping it while the guard is held would reenter it.
    pub(crate) fn pin_selected(
        &mut self,
        context: &WorkspaceContext,
    ) -> Result<PinnedCacheSource, CacheSourceFailure> {
        context
            .charge_metadata(
                fixed_bytes().ok_or_else(|| {
                    CacheSourceFailure::source(CacheSourceError::Overflow, context)
                })?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let count = self.blocks().count();
        let mut ids = context
            .metadata_vec(count)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        for block in self.blocks() {
            // CacheBlockId contains only fixed scalar/rank fields.
            ids.push(block.id().clone());
        }
        for (index, id) in ids.iter().enumerate() {
            if let Err(cause) = self.lifecycle.pin_source(id) {
                // No native owner or transfer was created. Undo only accepted
                // pins; metadata spending is cumulative and never refunded.
                for previous in &ids[..index] {
                    self.lifecycle
                        .release_source(previous)
                        .expect("accepted source pin remains canonical");
                }
                return Err(CacheSourceFailure::source(cause.into(), context));
            }
        }
        Ok(PinnedCacheSource {
            ids,
            generation: self.generation,
            selection: self.selection,
            manager: self.manager.clone(),
            _context: context.clone(),
        })
    }
}
impl Drop for PinnedCacheSource {
    fn drop(&mut self) {
        release_pins(&self.manager, self.ids.iter(), true);
    }
}
/// Same canonical retirement for whole source inventories, one source block,
/// and its completed demand lease. No reaping, promotion or native callback.
pub(super) fn release_pins<'a>(
    manager: &CacheResidencyManager,
    ids: impl Iterator<Item = &'a CacheBlockId>,
    source: bool,
) {
    for id in ids {
        let mut state = manager
            .inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let released = if source {
            state.lifecycle.release_source(id)
        } else {
            state.lifecycle.release(id)
        };
        debug_assert!(released.is_ok(), "source pin remains canonical until retirement");
        let retired = if released.is_ok() {
            // This does not create a discard request. Only a previously
            // completed original scan can have installed that exact request.
            super::super::original_discard::take_pending(&mut state, id).0
        } else {
            None
        };
        drop(state);
        drop(retired);
    }
}

fn fixed_bytes() -> Option<usize> {
    let parts = [
        size_of::<PinnedCacheSource>(),
        CacheResidencyManager::original_discard_control_bytes()?,
        size_of::<Result<PinnedCacheSource, CacheSourceFailure>>(),
        size_of::<Vec<CacheBlockId>>(),
        size_of::<CacheBlockId>(),
        size_of::<CacheLifecycleError>(),
        size_of::<Result<(), CacheLifecycleError>>(),
        size_of::<std::slice::Iter<'_, CacheBlockId>>(),
        size_of::<std::iter::Enumerate<std::slice::Iter<'_, CacheBlockId>>>(),
        size_of::<MutexGuard<'_, CacheManagerState>>(),
        size_of::<std::sync::LockResult<MutexGuard<'_, CacheManagerState>>>(),
        size_of::<(usize, usize, bool)>(),
        size_of::<WorkspaceContext>(),
        size_of::<CacheSourceFailure>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

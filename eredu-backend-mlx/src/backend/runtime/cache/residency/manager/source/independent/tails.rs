//! Exact source-tail claims consumed by the registered state-copy worker.
use super::*;
#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Ready,
    Spent,
    Published,
}
pub(super) struct CopyTail {
    layer: usize,
    tail: MutableCacheTail,
    status: Status,
}
impl CopyTail {
    pub(super) fn new(layer: usize, tail: MutableCacheTail) -> Self {
        Self {
            layer,
            tail,
            status: Status::Ready,
        }
    }
}
/// No caller-created scalar declaration can substitute for this source claim.
pub(crate) struct CopyTailClaim {
    manager: CacheResidencyManager,
    index: Option<usize>,
    layer: usize,
    tail: Option<MutableCacheTail>,
}
impl PreparedIndependentCacheManager {
    pub(crate) fn claim_copy_tail(
        &mut self,
        source: &CacheBlockSourceLoan<'_>,
        context: &WorkspaceContext,
    ) -> Result<CopyTailClaim, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        context
            .charge_metadata(tail_controls().ok_or_else(|| fail(CacheSourceError::Overflow))?)
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        if self.copy_reservation.is_none()
            || !self.matches_source(source.manager(), source.generation())
        {
            return Err(fail(CacheSourceError::Identity));
        }
        let layer = source.selection().layer();
        let index = self.copy_tails.iter().position(|tail| tail.layer == layer);
        let tail = source.tail();
        match index {
            Some(index) => {
                let expected = &mut self.copy_tails[index];
                if expected.status != Status::Ready || Some(expected.tail) != tail {
                    return Err(fail(CacheSourceError::Identity));
                }
                expected.status = Status::Spent;
            }
            None if tail.is_none() => {}
            None => return Err(fail(CacheSourceError::Identity)),
        }
        Ok(CopyTailClaim {
            manager: self.destination.clone(),
            index,
            layer,
            tail,
        })
    }
    pub(crate) fn publish_copy_tail(
        &mut self,
        claim: CopyTailClaim,
        arrays: Option<[&Array; 2]>,
        context: &WorkspaceContext,
    ) -> Result<(), CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        if !self.destination.same_catalog(&claim.manager) {
            return Err(fail(CacheSourceError::Identity));
        }
        let bytes = arrays
            .map_or(Some(0), |arrays| {
                arrays.into_iter().try_fold(0u64, |total, array| {
                    total.checked_add(u64::try_from(array.nbytes()).ok()?)
                })
            })
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        if claim.tail.map_or(0, |tail| tail.bytes) != bytes {
            return Err(fail(CacheSourceError::Geometry));
        }
        let Some(index) = claim.index else {
            if arrays.is_some() || claim.tail.is_some() {
                return Err(fail(CacheSourceError::Identity));
            }
            return Ok(());
        };
        let expected = self
            .copy_tails
            .get(index)
            .ok_or_else(|| fail(CacheSourceError::Identity))?;
        if expected.status != Status::Spent
            || expected.layer != claim.layer
            || Some(expected.tail) != claim.tail
        {
            return Err(fail(CacheSourceError::Identity));
        }
        {
            let mut state = self.destination.inner.state.try_lock().map_err(|cause| {
                fail(match cause {
                    TryLockError::WouldBlock => CacheSourceError::Busy,
                    TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
                })
            })?;
            if state.generation != self.installed.initial_generation()
                || state.lifecycle.tail(claim.layer).is_some()
            {
                return Err(fail(CacheSourceError::Identity));
            }
            state
                .lifecycle
                .set_tail_prepared(claim.layer, expected.tail)
                .map_err(|cause| fail(cause.into()))?;
        }
        self.publish_copy_occupancy(context)?;
        self.copy_tails[index].status = Status::Published;
        Ok(())
    }
    /// All actual slots and tails are complete before installation. The caller
    /// retains this same host owner beside any local state prefix until success.
    pub(crate) fn finish_copy(
        &mut self,
        host: &eredu_core::HostPreparationAuthority,
        context: &WorkspaceContext,
    ) -> Result<(), CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        context
            .charge_metadata(tail_controls().ok_or_else(|| fail(CacheSourceError::Overflow))?)
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        if self.copy_reservation.is_none()
            || self
                .copy_tails
                .iter()
                .any(|tail| tail.status != Status::Published)
        {
            return Err(fail(CacheSourceError::Identity));
        }
        {
            let mut state = self.destination.inner.state.try_lock().map_err(|cause| {
                fail(match cause {
                    TryLockError::WouldBlock => CacheSourceError::Busy,
                    TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
                })
            })?;
            if state.generation != self.installed.initial_generation()
                || state.lifecycle.catalog_population() != (self.copy_blocks, self.copy_tails.len())
                || state.blocks.len() != self.copy_blocks
                || state.original_copy_custody.is_some()
                || !self.destination.borrowed_storage_complete(&state)
            {
                return Err(fail(CacheSourceError::Identity));
            }
            state.original_copy_custody = Some(host.clone());
        }
        // Actual completed canonical arrays now retain their receipts. Only the
        // unused portion of the temporary occupancy reservation retires here.
        drop(self.copy_reservation.take());
        Ok(())
    }
}
pub(in super::super) fn tail_controls() -> Option<usize> {
    let frames = [
        size_of::<CopyTail>(),
        size_of::<CopyTailClaim>(),
        size_of::<Option<usize>>(),
        size_of::<(
            &mut PreparedIndependentCacheManager,
            &CacheBlockSourceLoan<'_>,
            &WorkspaceContext,
        )>(),
        size_of::<Option<[&Array; 2]>>(),
        size_of::<Option<MutableCacheTail>>(),
        size_of::<std::slice::Iter<'_, CopyTail>>(),
        size_of::<u64>(),
        size_of::<Option<u64>>(),
        size_of::<MutexGuard<'_, CacheManagerState>>(),
        size_of::<
            Result<
                MutexGuard<'_, CacheManagerState>,
                TryLockError<MutexGuard<'_, CacheManagerState>>,
            >,
        >(),
        size_of::<Result<CopyTailClaim, CacheSourceFailure>>(),
        size_of::<Result<(), CacheSourceFailure>>(),
        size_of::<eredu_core::HostPreparationAuthority>(),
        CacheBlockLifecycle::prepared_mutation_control_bytes()?,
        Array::descriptor_control_bytes()?.checked_mul(2)?,
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}

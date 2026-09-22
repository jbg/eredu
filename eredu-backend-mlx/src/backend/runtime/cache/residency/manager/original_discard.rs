//! Completed scan requests share canonical retirement and source-pin custody.
use super::*;
use crate::backend::nn::workspace::OriginalPagedDiscard;
use eredu_nn::workspace::HostMetadataFunding;
use safemlx::error::Exception;
use std::{mem::size_of, sync::TryLockError};

/// A privately constructed completed scan supplies its own execution proof.
/// The canonical removal and deferred final-pin worker are shared.
pub(crate) trait PreparedCacheDiscard {
    type Cause: From<CacheSourceError> + From<CacheResidencyError>;
    fn error(&self, cause: impl Into<Self::Cause>) -> Exception;
    fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception>;
    fn layer(&self) -> usize;
    fn ids(&self) -> &[CacheBlockId];
    fn frontier(&self) -> Option<(i64, i64)>;
    fn funding(&self) -> Option<HostMetadataFunding>;
}
impl PreparedCacheDiscard for OriginalPagedDiscard<'_, '_> {
    type Cause = crate::backend::nn::workspace::PagedMutationCause;
    fn error(&self, cause: impl Into<Self::Cause>) -> Exception {
        self.proof().error(cause)
    }
    fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception> {
        self.proof().validate_manager(manager, generation)
    }
    fn layer(&self) -> usize {
        self.proof().layer()
    }
    fn ids(&self) -> &[CacheBlockId] {
        self.ids()
    }
    fn frontier(&self) -> Option<(i64, i64)> {
        self.frontier()
    }
    fn funding(&self) -> Option<HostMetadataFunding> {
        self.proof().funding()
    }
}

/// This scalar request owns only its paid control lifetime, never a manager or
/// executable. Keeping it on a pinned record cannot form a manager/source cycle.
#[derive(Debug, Clone)]
pub(in super::super) struct PendingOriginalDiscard {
    visible_start: i64,
    prefix_tokens: i64,
    _funding: Option<HostMetadataFunding>,
}
impl CacheResidencyManager {
    pub(crate) fn discard_prepared<D: PreparedCacheDiscard>(
        &self,
        claim: &D,
    ) -> Result<(), Exception> {
        let Some((visible_start, prefix_tokens)) = claim.frontier() else {
            return Ok(());
        };
        if self.options().retains_discarded_for_persistence() {
            return Ok(());
        }
        for id in claim.ids() {
            let mut state = self.inner.state.try_lock().map_err(|cause| {
                claim.error(match cause {
                    TryLockError::WouldBlock => CacheSourceError::Busy,
                    TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
                })
            })?;
            claim.validate_manager(self, state.generation)?;
            if !self.borrowed_storage_complete(&state) {
                return Err(claim.error(CacheSourceError::PendingStorage));
            }
            state
                .blocks
                .validate_prepared_population(state.blocks.len())
                .map_err(|cause| claim.error(CacheSourceError::Lifecycle(cause.into())))?;
            if id.global_layer != claim.layer()
                || id.session_id != self.session_id()
                || id.representation != CacheRepresentation::KeyValue
                || !eredu_runtime::cache::CacheBlockSelection::outside_retained_window(
                    id.start,
                    id.end,
                    visible_start,
                    prefix_tokens,
                )
            {
                return Err(claim.error(CacheSourceError::Identity));
            }
            // An earlier completed scan may already have removed this exact
            // historical publication. No reconstructed identity is accepted.
            let Some(record) = state.blocks.get_mut(id) else {
                continue;
            };
            if record.imported {
                continue;
            }
            if record.original_discard.as_ref().is_some_and(|previous| {
                previous.prefix_tokens != prefix_tokens || previous.visible_start > visible_start
            }) {
                return Err(claim.error(CacheSourceError::Identity));
            }
            let previous = record.original_discard.replace(PendingOriginalDiscard {
                visible_start,
                prefix_tokens,
                _funding: claim.funding(),
            });
            let (retired, result) = take_pending(&mut state, id);
            // The manager generation and all worker epochs remain unchanged:
            // pending I/O was rejected above, and only completed exact IDs can
            // retire. Ordinary lifecycle cancellation is not entered here.
            drop(state);
            drop(retired);
            drop(previous);
            result.map_err(|cause| claim.error(cause))?;
        }
        Ok(())
    }
    pub(crate) fn prepared_discard_control_bytes<D: PreparedCacheDiscard>() -> Option<usize> {
        Self::discard_retirement_control_bytes()?
            .checked_add(size_of::<(&Self, &D)>())?
            .checked_add(size_of::<&[CacheBlockId]>())
    }
    pub(crate) fn discard_retirement_control_bytes() -> Option<usize> {
        let frames = [
            super::history::CacheHistoryNode::inspection_control_bytes()?,
            size_of::<PendingOriginalDiscard>(),
            size_of::<Option<PendingOriginalDiscard>>(),
            size_of::<(&mut CacheManagerState, &CacheBlockId)>(),
            size_of::<MutexGuard<'_, CacheManagerState>>(),
            size_of::<Result<MutexGuard<'_, CacheManagerState>, TryLockError<MutexGuard<'_, CacheManagerState>>>>(),
            size_of::<Option<CacheBlockRecord>>(),
            size_of::<(Option<CacheBlockRecord>, Result<(), CacheResidencyError>)>(),
            size_of::<Option<(i64, i64)>>(),
            size_of::<(i64, i64)>(),
            size_of::<std::slice::Iter<'_, CacheBlockId>>(),
            size_of::<std::slice::Iter<'_, Weak<CacheHistoryRetention>>>(),
            size_of::<Option<Arc<CacheHistoryRetention>>>(),
            size_of::<Result<(), Exception>>(),
            size_of::<Result<(), CacheLifecycleError>>(),
            size_of::<(&CacheManagerState, &CacheBlockId, &CacheBlockRecord, usize, CacheRepresentation, i64, i64)>(),
            eredu_runtime::cache::CacheRecordTable::<CacheBlockId, CacheBlockRecord>::mutation_control_bytes()?,
            CacheBlockLifecycle::prepared_mutation_control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}

/// Called only for an existing completed request, including at final pin release.
/// Every payload is returned even if final reporting fails, so it cannot retire
/// while the manager guard is held. A refused preflight leaves it canonical.
pub(super) fn take_pending(
    state: &mut CacheManagerState,
    id: &CacheBlockId,
) -> (Option<CacheBlockRecord>, Result<(), CacheResidencyError>) {
    let Some(record) = state.blocks.get(id) else {
        return (None, Ok(()));
    };
    let Some(request) = &record.original_discard else {
        return (None, Ok(()));
    };
    if record.pending_disk().is_some()
        || record.host_demotion_ticket().is_some()
        || !lifecycle::discard_candidate(
            state,
            id,
            record,
            id.global_layer,
            id.representation,
            request.visible_start,
            request.prefix_tokens,
        )
    {
        return (None, Ok(()));
    }
    if let Err(cause) = reporting::update_report_totals_prepared(state) {
        return (None, Err(cause));
    }
    let retired = match lifecycle::take_discarded_record(state, id) {
        Ok(record) => record,
        Err(cause) => return (None, Err(cause)),
    };
    // This only removes an already published contribution. On a pool reporting
    // failure the prior larger charge stays conservative; native storage still
    // travels back to the caller for retirement after unlock.
    let result = reporting::update_report_totals_prepared(state);
    (retired, result)
}

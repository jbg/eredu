//! Actual shared eviction policy; source facts alone never authorize a move.
use super::*;
use crate::backend::nn::workspace::OriginalPagedScanSource;
use safemlx::error::Exception;
use std::{mem::size_of, sync::TryLockError};
/// A lexical source minted by an admitted cache worker. Implementations keep
/// their own storage identity and failure custody; policy selection cannot
/// authenticate a source from a manager label or a byte count.
pub(crate) trait PreparedCacheTransferSource {
    type Cause: From<CacheSourceError> + From<CacheResidencyError>;
    fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception>;
    fn error(&self, cause: Self::Cause) -> Exception;
}
impl PreparedCacheTransferSource for OriginalPagedScanSource<'_> {
    type Cause = crate::backend::nn::workspace::PagedMutationCause;
    fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception> {
        OriginalPagedScanSource::validate_manager(self, manager, generation)
    }
    fn error(&self, cause: Self::Cause) -> Exception {
        OriginalPagedScanSource::error(self, cause)
    }
}
type Candidates<'a> = std::iter::FilterMap<
    eredu_runtime::cache::CacheRecordTableIter<'a, CacheBlockId, CacheBlockRecord>,
    fn((&'a CacheBlockId, &'a CacheBlockRecord)) -> Option<&'a CacheBlockId>,
>;
fn device_candidate<'a>(
    (id, row): (&'a CacheBlockId, &'a CacheBlockRecord),
) -> Option<&'a CacheBlockId> {
    (row.physical.phase() == CacheStoragePhase::Device).then_some(id)
}
fn unbacked_host_candidate<'a>(
    (id, row): (&'a CacheBlockId, &'a CacheBlockRecord),
) -> Option<&'a CacheBlockId> {
    (row.physical.phase() == CacheStoragePhase::HostUnbacked).then_some(id)
}
impl CacheResidencyManager {
    /// Positive delta is authenticated by the private append/load caller. This
    /// method only selects a current victim; the actual mover checks every pin.
    pub(crate) fn original_host_victim(
        &self,
        proof: &OriginalPagedScanSource<'_>,
        required: Option<&CacheBlockId>,
        additional: u64,
        tail: Option<(usize, u64)>,
        allow_recent: bool,
    ) -> Result<Option<CacheBlockId>, Exception> {
        self.prepared_host_victim(proof, required, additional, tail, allow_recent)
    }
    pub(crate) fn prepared_host_victim<P: PreparedCacheTransferSource>(
        &self,
        proof: &P,
        required: Option<&CacheBlockId>,
        additional: u64,
        tail: Option<(usize, u64)>,
        allow_recent: bool,
    ) -> Result<Option<CacheBlockId>, Exception> {
        let state = self.inner.state.try_lock().map_err(|cause| {
            proof.error(
                (match cause {
                    TryLockError::WouldBlock => CacheSourceError::Busy,
                    TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
                })
                .into(),
            )
        })?;
        proof.validate_manager(self, state.generation)?;
        if !self.borrowed_storage_complete(&state) {
            return Err(proof.error((CacheSourceError::PendingStorage).into()));
        }
        let mut bytes = state.telemetry.report.current_device_bytes;
        if let Some((layer, replacement)) = tail {
            let previous = state.lifecycle.tail(layer).map_or(0, |tail| tail.bytes);
            bytes = bytes
                .checked_sub(previous)
                .and_then(|n| n.checked_add(replacement))
                .ok_or_else(|| proof.error((CacheSourceError::Overflow).into()))?;
        }
        let bytes = bytes
            .checked_add(additional)
            .ok_or_else(|| proof.error((CacheSourceError::Overflow).into()))?;
        if bytes <= self.options().device_budget_bytes() {
            return Ok(None);
        }
        let candidates: Candidates<'_> = state
            .blocks
            .iter()
            .filter_map(device_candidate as fn(_) -> _);
        let choose = |recent| {
            state.lifecycle.device_eviction_candidate_borrowed(
                candidates.clone(),
                required,
                recent,
                self.options().eviction_policy(),
            )
        };
        let candidate = choose(self.options().recent_device_blocks())
            .map_err(|cause| proof.error((CacheSourceError::Lifecycle(cause)).into()))?;
        let candidate = match candidate {
            Some(id) => Some(id),
            None if allow_recent => choose(0)
                .map_err(|cause| proof.error((CacheSourceError::Lifecycle(cause)).into()))?,
            None => None,
        };
        candidate.cloned().map(Some).ok_or_else(|| {
            proof.error(
                (CacheResidencyError::BudgetExceeded {
                    tier: CacheTier::Device,
                    required: bytes,
                    budget: self.options().device_budget_bytes(),
                })
                .into(),
            )
        })
    }
    /// Shared tier-move selection for one actual durable write. Source pins
    /// remain owned; the writer independently validates their exact population.
    pub(crate) fn original_disk_write_victim(
        &self,
        proof: &OriginalPagedScanSource<'_>,
        required: Option<&CacheBlockId>,
        additional_host: u64,
        proactive: bool,
    ) -> Result<Option<CacheBlockId>, Exception> {
        self.prepared_disk_write_victim(proof, required, additional_host, proactive)
    }
    pub(crate) fn prepared_disk_write_victim<P: PreparedCacheTransferSource>(
        &self,
        proof: &P,
        required: Option<&CacheBlockId>,
        additional_host: u64,
        proactive: bool,
    ) -> Result<Option<CacheBlockId>, Exception> {
        if !matches!(
            self.options().live_disk_policy(),
            LiveCacheDiskPolicy::Enabled { .. }
        ) {
            return Ok(None);
        }
        let state = self.inner.state.try_lock().map_err(|cause| {
            proof.error(
                (match cause {
                    TryLockError::WouldBlock => CacheSourceError::Busy,
                    TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
                })
                .into(),
            )
        })?;
        proof.validate_manager(self, state.generation)?;
        if !self.borrowed_storage_complete(&state) {
            return Err(proof.error((CacheSourceError::PendingStorage).into()));
        }
        let current = state.telemetry.report.current_host_bytes;
        let needed = current
            .checked_add(additional_host)
            .ok_or_else(|| proof.error((CacheSourceError::Overflow).into()))?;
        let budget = self.options().host_budget_bytes();
        if needed <= budget && !(proactive && current != 0 && current >= budget) {
            return Ok(None);
        }
        let candidates: Candidates<'_> = state
            .blocks
            .iter()
            .filter_map(unbacked_host_candidate as fn(_) -> _);
        let chosen = state
            .lifecycle
            .tier_move_candidate_borrowed(candidates, required, 0, self.options().eviction_policy())
            .map_err(|cause| proof.error((CacheSourceError::Lifecycle(cause)).into()))?;
        if let Some(id) = chosen {
            return Ok(Some(id.clone()));
        }
        if needed <= budget {
            return Ok(None);
        }
        Err(proof.error(
            (CacheResidencyError::BudgetExceeded {
                tier: CacheTier::Host,
                required: needed,
                budget,
            })
            .into(),
        ))
    }
    pub(crate) fn with_original_host_source<R>(
        &self,
        proof: &OriginalPagedScanSource<'_>,
        id: &CacheBlockId,
        run: impl FnOnce(&mut source::CacheBlockSourceLoan<'_>) -> Result<R, Exception>,
    ) -> Result<R, Exception> {
        let selection = eredu_runtime::CacheBlockSelection::new(
            id.global_layer,
            id.representation,
            id.start,
            id.end,
            0,
        );
        self.with_source_loan_inner(
            selection,
            Some(proof.publication_controls()),
            |cause| proof.error(cause),
            |mut loan| {
                proof.validate_loan(&loan)?;
                run(&mut loan)
            },
        )
    }
    pub(crate) fn original_host_policy_control_bytes() -> Option<usize> {
        let wrappers = [
            size_of::<(
                &Self,
                &OriginalPagedScanSource<'_>,
                Option<&CacheBlockId>,
                u64,
                Option<(usize, u64)>,
                bool,
            )>(),
            size_of::<(
                &Self,
                &OriginalPagedScanSource<'_>,
                Option<&CacheBlockId>,
                u64,
                bool,
            )>(),
        ];
        Self::prepared_host_policy_control_bytes::<OriginalPagedScanSource<'_>>()?.checked_add(
            wrappers
                .into_iter()
                .try_fold(std::mem::size_of_val(&wrappers), usize::checked_add)?,
        )
    }
    pub(crate) fn prepared_host_policy_control_bytes<P: PreparedCacheTransferSource>()
    -> Option<usize> {
        let frames = [
            size_of::<(
                &Self,
                &P,
                Option<&CacheBlockId>,
                u64,
                Option<(usize, u64)>,
                bool,
            )>(),
            size_of::<Candidates<'_>>(),
            size_of::<CacheBlockId>(),
            size_of::<Option<CacheBlockId>>(),
            size_of::<Result<Option<CacheBlockId>, Exception>>(),
            size_of::<MutexGuard<'_, CacheManagerState>>(),
            size_of::<
                Result<
                    MutexGuard<'_, CacheManagerState>,
                    TryLockError<MutexGuard<'_, CacheManagerState>>,
                >,
            >(),
            size_of::<(u64, u64, usize)>(),
            size_of::<(&Self, &P, Option<&CacheBlockId>, u64, bool)>(),
            size_of::<(u64, u64, u64, bool)>(),
            CacheBlockLifecycle::eviction_control_bytes::<Candidates<'_>>()?.checked_mul(2)?,
            size_of::<(
                &Self,
                &CacheManagerState,
                &Candidates<'_>,
                Option<&CacheBlockId>,
                usize,
            )>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}

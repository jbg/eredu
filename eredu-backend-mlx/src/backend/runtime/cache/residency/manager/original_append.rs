//! Canonical publication shared by authentic Original and ordinary source loans.
use super::prepared_append::PreparedAppend;
use super::*;
use crate::backend::nn::workspace::{OrdinaryPagedAppend, OriginalPagedAppendClaim};
use safemlx::error::Exception;
use std::{mem::size_of, sync::TryLockError};

impl CacheResidencyManager {
    fn prepared_append_lock<'a, C: PreparedAppend>(
        &'a self,
        claim: &C,
    ) -> Result<MutexGuard<'a, CacheManagerState>, Exception> {
        let state = self.inner.state.try_lock().map_err(|cause| {
            claim.error(match cause {
                TryLockError::WouldBlock => CacheSourceError::Busy,
                TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
            })
        })?;
        claim.validate_manager(self, state.generation)?;
        if !self.borrowed_storage_complete(&state) {
            return Err(claim.error(CacheSourceError::PendingStorage));
        }
        for (id, record) in state.blocks.iter() {
            if id.global_layer == claim.layer()
                && record.physical.phase() != CacheStoragePhase::Device
            {
                claim.validate_retained_storage(
                    id,
                    record.physical.phase(),
                    record.disk().and_then(DiskLocation::file_source).as_ref(),
                )?;
            }
        }
        if state
            .blocks
            .iter()
            .filter(|(id, record)| {
                id.global_layer == claim.layer()
                    && record.physical.phase() == CacheStoragePhase::Device
            })
            .any(|(_, record)| {
                record.physical.device_resource().is_none_or(|arrays| {
                    arrays.arrays().iter().any(|array| {
                        CacheBlockMetadata::floating_dtype_bytes(array.dtype()).is_none()
                    })
                })
            })
        {
            return Err(claim.error(CacheSourceError::Geometry));
        }
        Ok(state)
    }
    /// No native work occurs under this guard. Source/role authentication
    /// precedes the same actual manager transfer-device and tail agreement.
    pub(crate) fn begin_prepared_append<C: PreparedAppend>(
        &self,
        claim: &C,
        tail_bytes: u64,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let snapshot =
            safemlx::StreamCopyPlan::<()>::capture(stream).map_err(|cause| claim.error(cause))?;
        let device = CacheTransferDevice {
            device_type: snapshot.device_type(),
            index: snapshot.device_index(),
        };
        let mut state = self.prepared_append_lock(claim)?;
        if state.transfer_device.is_some_and(|bound| bound != device) {
            return Err(claim.error(CacheSourceError::Identity));
        }
        let (tail_start, _, offset) = claim.plan().initial_frontier();
        match state.lifecycle.tail(claim.layer()) {
            Some(tail) if tail.bytes == tail_bytes && tail.end == offset => {}
            None if tail_bytes == 0 && tail_start == offset => {}
            _ => return Err(claim.error(CacheSourceError::Identity)),
        }
        claim.validate_rollback_tail(tail_bytes, offset)?;
        state.transfer_device = Some(device);
        Ok(())
    }
    pub(crate) fn publish_prepared_tail<C: PreparedAppend>(
        &self,
        claim: &mut C,
        bytes: u64,
        end: i64,
        clearing: bool,
        restoring: bool,
    ) -> Result<(), Exception> {
        claim.validate_tail_update(bytes, end, clearing, restoring)?;
        let mut state = self.prepared_append_lock(claim)?;
        let previous = state
            .lifecycle
            .set_tail_prepared(claim.layer(), MutableCacheTail { bytes, end })
            .map_err(|cause| claim.error(CacheResidencyError::from(cause)))?;
        let allocated = previous.is_none_or(|tail| tail.bytes == 0) && bytes > 0;
        if allocated {
            state.telemetry.report.tail_allocations += 1;
        }
        if let Err(cause) = device_fit(&mut state, self.options(), None) {
            state.lifecycle.restore_tail(claim.layer(), previous);
            if allocated {
                state.telemetry.report.tail_allocations =
                    state.telemetry.report.tail_allocations.saturating_sub(1);
            }
            let restored = reporting::update_report_totals_prepared(&mut state);
            drop(state);
            return Err(match restored {
                Ok(()) => claim.error(cause),
                Err(rollback) => rollback_error(claim.error(cause), claim.error(rollback)),
            });
        }
        drop(state);
        claim.did_update_tail(clearing, restoring);
        Ok(())
    }
    /// Exact arrays settle outside all manager/source loans before a native
    /// record enters the same canonical logical/physical insertion worker.
    pub(crate) fn seal_prepared_block<C: PreparedAppend>(
        &self,
        claim: &mut C,
        id: CacheBlockId,
        arrays: CacheBlockArrays,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let bytes = arrays
            .arrays()
            .into_iter()
            .try_fold(0u64, |n, array| n.checked_add(array.nbytes() as u64))
            .ok_or_else(|| claim.error(CacheSourceError::Overflow))?;
        claim.rebalance_host(bytes, None, Some(&id), stream)?;
        let (metadata, protected) = claim.take_publication(&id, &arrays)?;
        claim.complete(arrays.arrays(), stream)?;
        let record = metadata.into_record(id.clone(), arrays, false);
        let mut state = match self.prepared_append_lock(claim) {
            Ok(state) => state,
            Err(cause) => {
                drop(record);
                return Err(cause);
            }
        };
        if let Err((cause, record)) =
            publication::insert_record(&mut state, id.clone(), record, protected, true)
        {
            drop(state);
            drop(record);
            return Err(claim.error(cause));
        }
        if let Err(cause) = device_fit(&mut state, self.options(), Some(&id)) {
            // No worker or lease can have acquired this unpublished scope's
            // newly inserted record while this same guard remains held.
            let removal = state
                .lifecycle
                .remove(&id)
                .map_err(CacheResidencyError::from);
            let removed = if removal.is_ok() {
                state.blocks.remove(&id)
            } else {
                None
            };
            let restored = reporting::update_report_totals_prepared(&mut state);
            drop(state);
            drop(removed);
            let rollback = removal.and(restored);
            return Err(match rollback {
                Ok(()) => claim.error(cause),
                Err(rollback) => rollback_error(claim.error(cause), claim.error(rollback)),
            });
        }
        state.telemetry.report.block_seals += 1;
        drop(state);
        claim.published(&id)
    }
    pub(crate) fn rollback_prepared_append<C: PreparedAppend>(
        &self,
        claim: &C,
        tail_bytes: u64,
        offset: i64,
    ) -> Result<(), Exception> {
        claim.validate_rollback_tail(tail_bytes, offset)?;
        let mut state = self.prepared_append_lock(claim)?;
        // Check all leases before any removal. IDs come only from this exact
        // consumed publication program; existing source blocks are untouched.
        for index in 0..claim.published_count() {
            let id = claim
                .publication_id(index)
                .expect("published program index");
            if !state.blocks.contains_key(id) {
                return Err(claim.error(CacheSourceError::Identity));
            }
            if state
                .lifecycle
                .is_leased(id)
                .map_err(|cause| claim.error(CacheResidencyError::from(cause)))?
            {
                return Err(claim.error(CacheResidencyError::from(
                    CacheLifecycleError::BlockLeased(id.clone()),
                )));
            }
        }
        let generation = state
            .generation
            .checked_add(
                u64::try_from(claim.published_count())
                    .map_err(|_| claim.error(CacheSourceError::Overflow))?,
            )
            .ok_or_else(|| claim.error(CacheSourceError::Overflow))?;
        // Remove outside each guard so native arrays/account metadata never
        // retire beneath the manager mutex. The source role is already fenced
        // if an operation fails, and no worker exists on this path.
        for index in (0..claim.published_count()).rev() {
            let id = claim
                .publication_id(index)
                .expect("published program index");
            let removed = lifecycle::take_unleased_record(&mut state, id)
                .map_err(|cause| claim.error(cause))?;
            drop(state);
            drop(removed);
            state = self.prepared_append_lock(claim)?;
        }
        state
            .lifecycle
            .set_tail_prepared(
                claim.layer(),
                MutableCacheTail {
                    bytes: tail_bytes,
                    end: offset,
                },
            )
            .map_err(|cause| claim.error(CacheResidencyError::from(cause)))?;
        state.generation = generation;
        reporting::update_report_totals_prepared(&mut state).map_err(|cause| claim.error(cause))
    }
    pub(crate) fn prepared_append_rollback_error(
        primary: Exception,
        rollback: Exception,
    ) -> Exception {
        rollback_error(primary, rollback)
    }
    pub(crate) fn original_append_control_bytes() -> Option<usize> {
        Self::prepared_append_control_bytes::<OriginalPagedAppendClaim<'_>>()
    }
    pub(crate) fn ordinary_append_control_bytes() -> Option<usize> {
        Self::prepared_append_control_bytes::<OrdinaryPagedAppend>()
    }
    fn prepared_append_control_bytes<C: PreparedAppend>() -> Option<usize> {
        let frames = [
            size_of::<(&Self, &C, &Stream)>(),
            size_of::<MutexGuard<'_, CacheManagerState>>(),
            size_of::<
                Result<
                    MutexGuard<'_, CacheManagerState>,
                    TryLockError<MutexGuard<'_, CacheManagerState>>,
                >,
            >(),
            size_of::<CacheTransferDevice>(),
            size_of::<safemlx::StreamCopyPlan<()>>(),
            size_of::<Result<safemlx::StreamCopyPlan<()>, safemlx::StreamCopyCause>>(),
            size_of::<CacheBlockId>(),
            size_of::<CacheBlockArrays>(),
            size_of::<CacheBlockRecord>(),
            size_of::<Option<CacheFileSource>>(),
            size_of::<Option<&CacheFileSource>>(),
            size_of::<Option<CacheBlockRecord>>(),
            size_of::<Option<MutableCacheTail>>(),
            size_of::<Result<(), Exception>>(),
            size_of::<Result<(), CacheResidencyError>>(),
            size_of::<Result<(), (CacheResidencyError, CacheBlockRecord)>>(),
            size_of::<transitions::BudgetSnapshot>(),
            size_of::<(u64, usize, bool)>(),
            Exception::retained_source_control_bytes::<RollbackFailure>()?,
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}

/// Same local/pool boundary as ordinary rebalance. Source-specific transfer
/// preparation precedes this common publication check; failures roll back.
pub(super) fn device_fit(
    state: &mut CacheManagerState,
    options: &PagedCacheOptions,
    required: Option<&CacheBlockId>,
) -> Result<(), CacheResidencyError> {
    reporting::update_report_totals_prepared(state)?;
    let budget = transitions::budget_snapshot(state, options)?;
    let cause = if budget.local_device_over || budget.pool_device_over {
        Some(if budget.pool_device_over && !budget.local_device_over {
            CachePoolError::BudgetExceeded {
                resource: CachePoolResource::Device,
                required: budget.pool.current_device_bytes,
                budget: budget.pool.limits.device_bytes(),
            }
            .into()
        } else {
            CacheResidencyError::BudgetExceeded {
                tier: CacheTier::Device,
                required: state.telemetry.report.current_device_bytes,
                budget: options.device_budget_bytes(),
            }
        })
    } else if budget.local_host_over || budget.pool_host_over {
        Some(if budget.pool_host_over && !budget.local_host_over {
            CachePoolError::BudgetExceeded {
                resource: CachePoolResource::Host,
                required: budget.pool.current_host_bytes,
                budget: budget.pool.limits.host_bytes(),
            }
            .into()
        } else {
            CacheResidencyError::BudgetExceeded {
                tier: CacheTier::Host,
                required: state.telemetry.report.current_host_bytes,
                budget: options.host_budget_bytes(),
            }
        })
    } else {
        None
    };
    if let Some(cause) = cause {
        state.telemetry.report.failures += 1;
        if let Some(id) = required {
            state.layer_activity_mut(id.global_layer).failures += 1;
        } else {
            state.telemetry.unassigned_activity_mut().failures += 1;
        }
        Err(cause)
    } else {
        Ok(())
    }
}
#[derive(Debug, thiserror::Error)]
#[error("{primary}; additionally failed to restore paged append: {rollback}")]
struct RollbackFailure {
    #[source]
    primary: Exception,
    rollback: Exception,
}
pub(super) fn rollback_error(primary: Exception, rollback: Exception) -> Exception {
    Exception::from_retained_source(RollbackFailure { primary, rollback })
}

pub(super) fn rollback_control_bytes() -> Option<usize> {
    Exception::retained_source_control_bytes::<RollbackFailure>()
}

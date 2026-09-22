//! Shared completed canonical publication for first stores and immutable reuse.
use super::*;
/// Source checks for the shared canonical transition. Implementations are
/// minted by their actual admitted source; values and pin counts alone are not
/// authority to replace a record or release its device reservation.
pub(crate) trait PreparedHostEviction {
    type Cause: From<CacheSourceError> + From<CacheResidencyError>;
    fn id(&self) -> &CacheBlockId;
    fn arrays(&self) -> [&Array; 2];
    fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception>;
    fn validate_pin_counts(
        &self,
        lifecycle: &eredu_runtime::CacheBlockLifecycle,
    ) -> Result<(), Exception>;
    fn error(&self, cause: Self::Cause) -> Exception;
}
impl PreparedHostEviction for OriginalPagedHostEviction<'_, '_> {
    type Cause = crate::backend::nn::workspace::PagedMutationCause;
    fn id(&self) -> &CacheBlockId {
        OriginalPagedHostEviction::id(self)
    }
    fn arrays(&self) -> [&Array; 2] {
        OriginalPagedHostEviction::arrays(self)
    }
    fn validate_manager(
        &self,
        manager: &CacheResidencyManager,
        generation: u64,
    ) -> Result<(), Exception> {
        self.source().validate_manager(manager, generation)
    }
    fn validate_pin_counts(
        &self,
        lifecycle: &eredu_runtime::CacheBlockLifecycle,
    ) -> Result<(), Exception> {
        OriginalPagedHostEviction::validate_pin_counts(self, lifecycle)
    }
    fn error(&self, cause: Self::Cause) -> Exception {
        self.source().error(cause)
    }
}
pub(crate) fn commit_host<P: PreparedHostEviction>(
    manager: &CacheResidencyManager,
    proof: &P,
    host: HostCacheBlock,
    capacity: u64,
    reservation: &mut CachePoolReservation,
    copied: bool,
) -> Result<eredu_runtime::cache::CacheDeviceDemotion<CacheBlockArrays>, Exception> {
    let mut state = manager.inner.state.try_lock().map_err(|cause| {
        proof.error(
            (match cause {
                TryLockError::WouldBlock => CacheSourceError::Busy,
                TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
            })
            .into(),
        )
    })?;
    validate_source(manager, proof, &state)?;
    reporting::update_report_totals_prepared(&mut state)
        .map_err(|cause| proof.error(cause.into()))?;
    if state
        .telemetry
        .report
        .current_host_bytes
        .checked_add(capacity)
        .is_none_or(|n| n > manager.options().host_budget_bytes())
    {
        return Err(proof.error(CacheSourceError::PromotionRequired.into()));
    }
    let bytes = state
        .blocks
        .get(proof.id())
        .expect("validated record")
        .bytes;
    let transferred = if copied { bytes } else { 0 };
    let overflow = || proof.error(CacheSourceError::Overflow.into());
    let host_demotions = state
        .telemetry
        .report
        .host_demotions
        .checked_add(1)
        .ok_or_else(overflow)?;
    let transfer_bytes = state
        .telemetry
        .report
        .transfer_bytes
        .checked_add(transferred)
        .ok_or_else(overflow)?;
    let activity = state.layer_activity_mut(proof.id().global_layer);
    let layer_demotions = activity
        .host_demotions
        .checked_add(1)
        .ok_or_else(overflow)?;
    let layer_transfer_bytes = activity
        .transfer_bytes
        .checked_add(transferred)
        .ok_or_else(overflow)?;
    let replaced = state
        .blocks
        .get_mut(proof.id())
        .expect("validated record")
        .physical
        .demote_completed(host)
        .expect("same locked stable unbacked Device phase");
    if let Err(cause) = reporting::update_report_totals_prepared_replacement(
        &mut state,
        reservation,
        &manager.inner.pool_membership,
    ) {
        let host = state
            .blocks
            .get_mut(proof.id())
            .expect("same locked record")
            .physical
            .restore_device(replaced)
            .expect("same locked completed demotion");
        reporting::update_report_totals(&mut state);
        drop(state);
        drop(host);
        return Err(proof.error(cause.into()));
    }
    state.telemetry.report.host_demotions = host_demotions;
    state.telemetry.report.transfer_bytes = transfer_bytes;
    let activity = state.layer_activity_mut(proof.id().global_layer);
    activity.host_demotions = layer_demotions;
    activity.transfer_bytes = layer_transfer_bytes;
    Ok(replaced)
}

/// Authenticate the actual canonical pair and owned pin population before any
/// ordinary copy is submitted. Publication repeats this check under its lock.
pub(super) fn preflight<P: PreparedHostEviction>(
    manager: &CacheResidencyManager,
    proof: &P,
    capacity: u64,
) -> Result<(), Exception> {
    let mut state = manager.inner.state.try_lock().map_err(|cause| {
        proof.error(
            match cause {
                TryLockError::WouldBlock => CacheSourceError::Busy,
                TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
            }
            .into(),
        )
    })?;
    validate_source(manager, proof, &state)?;
    reporting::update_report_totals_prepared(&mut state)
        .map_err(|cause| proof.error(cause.into()))?;
    if state
        .telemetry
        .report
        .current_host_bytes
        .checked_add(capacity)
        .is_none_or(|bytes| bytes > manager.options().host_budget_bytes())
    {
        return Err(proof.error(CacheSourceError::PromotionRequired.into()));
    }
    Ok(())
}
fn validate_source<P: PreparedHostEviction>(
    manager: &CacheResidencyManager,
    proof: &P,
    state: &CacheManagerState,
) -> Result<(), Exception> {
    proof.validate_manager(manager, state.generation)?;
    if !manager.borrowed_storage_complete(&state) {
        return Err(proof.error(CacheSourceError::PendingStorage.into()));
    }
    proof.validate_pin_counts(&state.lifecycle)?;
    let record = state
        .blocks
        .get(proof.id())
        .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?;
    if record.physical.phase() != CacheStoragePhase::Device || record.disk().is_some() {
        return Err(proof.error(CacheSourceError::Identity.into()));
    }
    let actual = record
        .physical
        .device_resource()
        .ok_or_else(|| proof.error(CacheSourceError::Identity.into()))?
        .arrays();
    for (actual, expected) in actual.into_iter().zip(proof.arrays()) {
        let allocation = actual
            .try_allocation_info()
            .map_err(|cause| proof.error(CacheSourceError::ArrayMetadata(cause).into()))?;
        let expected = expected
            .try_allocation_info()
            .map_err(|cause| proof.error(CacheSourceError::ArrayMetadata(cause).into()))?;
        if allocation.is_none() || allocation != expected {
            return Err(proof.error(CacheSourceError::Identity.into()));
        }
    }
    Ok(())
}
pub(crate) fn control_bytes() -> Option<usize> {
    prepared_control_bytes::<OriginalPagedHostEviction<'_, '_>>()
}
pub(crate) fn prepared_control_bytes<P: PreparedHostEviction>() -> Option<usize> {
    let frames = [
        size_of::<(&CacheResidencyManager, &P, &CacheManagerState)>(),
        size_of::<(&CacheResidencyManager, &P, u64)>(),
        size_of::<Result<(), Exception>>(),
        size_of::<(
            &CacheResidencyManager,
            &P,
            HostCacheBlock,
            u64,
            &mut CachePoolReservation,
            bool,
        )>(),
        size_of::<Result<eredu_runtime::cache::CacheDeviceDemotion<CacheBlockArrays>, Exception>>(),
        size_of::<[&Array; 2]>(),
        size_of::<std::iter::Zip<std::array::IntoIter<&Array, 2>, std::array::IntoIter<&Array, 2>>>(
        ),
        size_of::<(Option<AllocationInfo>, Option<AllocationInfo>, u64, u64)>(),
        size_of::<HostCacheBlock>(),
        size_of::<[u64; 4]>(),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}

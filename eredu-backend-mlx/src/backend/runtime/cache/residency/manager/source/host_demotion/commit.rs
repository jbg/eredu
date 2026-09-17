//! Shared completed canonical publication for first stores and immutable reuse.
use super::*;
pub(crate) fn commit_host(
    manager: &CacheResidencyManager,
    proof: &OriginalPagedHostEviction<'_, '_>,
    host: HostCacheBlock,
    capacity: u64,
    reservation: &mut CachePoolReservation,
    copied: bool,
) -> Result<eredu_runtime::cache::CacheDeviceDemotion<CacheBlockArrays>, Exception> {
    let source = proof.source();
    let mut state = manager.inner.state.try_lock().map_err(|cause| {
        source.error(match cause {
            TryLockError::WouldBlock => CacheSourceError::Busy,
            TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
        })
    })?;
    source.validate_manager(manager, state.generation)?;
    if !manager.borrowed_storage_complete(&state) {
        return Err(source.error(CacheSourceError::PendingStorage));
    }
    proof.validate_pin_counts(&state.lifecycle)?;
    let record = state
        .blocks
        .get(proof.id())
        .ok_or_else(|| source.error(CacheSourceError::Identity))?;
    if record.physical.phase() != CacheStoragePhase::Device || record.disk().is_some() {
        return Err(source.error(CacheSourceError::Identity));
    }
    let actual = record
        .physical
        .device_resource()
        .ok_or_else(|| source.error(CacheSourceError::Identity))?
        .arrays();
    for (actual, expected) in actual.into_iter().zip(proof.arrays()) {
        let allocation = actual
            .try_allocation_info()
            .map_err(|cause| source.error(CacheSourceError::ArrayMetadata(cause)))?;
        let expected = expected
            .try_allocation_info()
            .map_err(|cause| source.error(CacheSourceError::ArrayMetadata(cause)))?;
        if allocation.is_none() || allocation != expected {
            return Err(source.error(CacheSourceError::Identity));
        }
    }
    reporting::update_report_totals_prepared(&mut state).map_err(|cause| source.error(cause))?;
    if state
        .telemetry
        .report
        .current_host_bytes
        .checked_add(capacity)
        .is_none_or(|n| n > manager.options().host_budget_bytes())
    {
        return Err(source.error(CacheSourceError::PromotionRequired));
    }
    let bytes = state
        .blocks
        .get(proof.id())
        .expect("validated record")
        .bytes;
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
        return Err(source.error(cause));
    }
    state.telemetry.report.host_demotions += 1;
    let transferred = if copied { bytes } else { 0 };
    state.telemetry.report.transfer_bytes += transferred;
    let activity = state.layer_activity_mut(proof.id().global_layer);
    activity.host_demotions += 1;
    activity.transfer_bytes += transferred;
    Ok(replaced)
}

pub(crate) fn control_bytes() -> Option<usize> {
    let frames = [
        size_of::<(
            &CacheResidencyManager,
            &OriginalPagedHostEviction<'_, '_>,
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
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}

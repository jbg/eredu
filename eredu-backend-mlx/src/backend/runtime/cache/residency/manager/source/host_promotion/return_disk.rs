//! Completed backed Device storage returns to its exact durable source.
use super::*;
use crate::backend::nn::workspace::OriginalPagedHostReturn;
impl PreparedCacheHostPromotion {
    pub(crate) fn has_backing(&self) -> bool {
        self.backing.is_some()
    }
    pub(crate) fn return_to_disk(
        &mut self,
        proof: &OriginalPagedHostReturn<'_, '_>,
    ) -> Result<(), Exception> {
        let source = proof.source();
        if !self.published
            || !self.completed
            || self.demoted
            || self.replaced.is_some()
            || self.disk_replaced.is_some()
            || proof.id() != &self.id
        {
            return Err(source.error(CacheSourceError::Identity));
        }
        self.validate_host_source(source)?;
        let backing = self
            .backing
            .as_ref()
            .ok_or_else(|| source.error(CacheSourceError::Identity))?;
        let arrays = [
            self.outputs[0].as_ref().expect("completed first"),
            self.outputs[1].as_ref().expect("completed second"),
        ];
        let device = return_device(
            &self.manager,
            &self.id,
            backing,
            arrays,
            self.reservation.as_mut().expect("completed promotion reservation"),
            proof,
        )?;
        self.disk_replaced = Some(device);
        self.demoted = true;
        Ok(())
    }
}
pub(super) fn return_device(
    manager: &CacheResidencyManager,
    id: &CacheBlockId,
    backing: &DiskLocation,
    arrays: [&Array; 2],
    reservation: &mut CachePoolReservation,
    proof: &OriginalPagedHostReturn<'_, '_>,
) -> Result<CacheBlockArrays, Exception> {
    let source = proof.source();
    if proof.id() != id {
        return Err(source.error(CacheSourceError::Identity));
    }
    let file = backing
        .live_source
        .as_ref()
        .ok_or_else(|| source.error(CacheSourceError::Identity))?;
    for array in arrays {
        source.observer().validate_completed_array(array)?;
    }
    let permission = proof.bind(arrays);
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
    permission.validate_pin_counts(&state.lifecycle)?;
    let record = state
        .blocks
        .get(id)
        .ok_or_else(|| source.error(CacheSourceError::Identity))?;
    if record.physical.phase() != CacheStoragePhase::Device
        || record
            .disk()
            .and_then(|value| value.live_source.as_ref())
            .is_none_or(|value| !value.same_source(file))
    {
        return Err(source.error(CacheSourceError::Identity));
    }
    let actual = record
        .physical
        .device_resource()
        .ok_or_else(|| source.error(CacheSourceError::Identity))?
        .arrays();
    for (actual, expected) in actual.into_iter().zip(arrays) {
        let a = actual
            .try_allocation_info()
            .map_err(|cause| source.error(CacheSourceError::ArrayMetadata(cause)))?;
        let b = expected
            .try_allocation_info()
            .map_err(|cause| source.error(CacheSourceError::ArrayMetadata(cause)))?;
        if a.is_none() || a != b {
            return Err(source.error(CacheSourceError::Identity));
        }
    }
    reporting::update_report_totals_prepared(&mut state).map_err(|cause| source.error(cause))?;
    let device = state
        .blocks
        .get_mut(id)
        .expect("validated Device source")
        .physical
        .release_device_to_disk()
        .expect("same locked backed Device source");
    if let Err(cause) = reporting::update_report_totals_prepared_replacement(
        &mut state,
        reservation,
        &manager.inner.pool_membership,
    ) {
        let previous = std::mem::replace(
            &mut state.blocks.get_mut(id).expect("same record").physical,
            MlxCacheBlockStorage::device(id.clone(), device, Some(backing.clone())),
        );
        let _ = reporting::update_report_totals_prepared(&mut state);
        drop(state);
        drop(previous);
        return Err(source.error(cause));
    }
    state.telemetry.report.disk_demotions += 1;
    state.layer_activity_mut(id.global_layer).disk_demotions += 1;
    drop(state);
    Ok(device)
}

pub(super) fn control_bytes() -> Option<usize> {
    let frames = [
        crate::backend::nn::workspace::OriginalPagedHostEviction::control_bytes()?,
        reporting::report_query_control_bytes()?,
        size_of::<MutexGuard<'_, CacheManagerState>>(),
        size_of::<
            Result<
                MutexGuard<'_, CacheManagerState>,
                TryLockError<MutexGuard<'_, CacheManagerState>>,
            >,
        >(),
        size_of::<eredu_runtime::cache::CacheRecordTableIter<'_, CacheBlockId, CacheBlockRecord>>(),
        size_of::<Result<Option<safemlx::ArrayAllocationInfo>, safemlx::ArrayMetadataError>>(),
        size_of::<(
            &mut PreparedCacheHostPromotion,
            &OriginalPagedHostReturn<'_, '_>,
        )>(),
        size_of::<Option<CacheBlockArrays>>(),
        size_of::<(
            &CacheResidencyManager,
            &CacheBlockId,
            &DiskLocation,
            [&Array; 2],
            &mut CachePoolReservation,
            &OriginalPagedHostReturn<'_, '_>,
        )>(),
        size_of::<Result<CacheBlockArrays, Exception>>(),
        size_of::<CacheBlockArrays>(),
        size_of::<MlxCacheBlockStorage>(),
        size_of::<Option<DiskLocation>>(),
        size_of::<(&DiskLocation, &LiveCacheBlockSource)>(),
        size_of::<Result<(), Exception>>(),
        size_of::<[&Array; 2]>(),
        size_of::<std::iter::Zip<std::array::IntoIter<&Array, 2>, std::array::IntoIter<&Array, 2>>>(
        ),
        size_of::<(
            Option<safemlx::ArrayAllocationInfo>,
            Option<safemlx::ArrayAllocationInfo>,
        )>(),
    ];
    frames
        .into_iter()
        .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
}

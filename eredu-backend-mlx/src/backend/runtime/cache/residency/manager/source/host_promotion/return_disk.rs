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
            self.reservation
                .as_mut()
                .expect("completed promotion reservation"),
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
    for array in arrays {
        source.observer().validate_completed_array(array)?;
    }
    let permission = proof.bind(arrays);
    return_device_prepared(manager, id, backing, reservation, &permission)
}

/// One canonical return worker for authenticated completed Device owners.
/// The proof supplies the actual arrays and pin inventory, independently of
/// whether their selected execution uses an ordinary or original scope.
pub(super) fn return_device_prepared<P: super::super::host_demotion::PreparedHostEviction>(
    manager: &CacheResidencyManager,
    id: &CacheBlockId,
    backing: &DiskLocation,
    reservation: &mut CachePoolReservation,
    proof: &P,
) -> Result<CacheBlockArrays, Exception> {
    let fail = |cause: CacheSourceError| proof.error(cause.into());
    let arrays = proof.arrays();
    if proof.id() != id {
        return Err(fail(CacheSourceError::Identity));
    }
    let file = backing
        .file_source()
        .ok_or_else(|| fail(CacheSourceError::Identity))?;
    let mut state = manager.inner.state.try_lock().map_err(|cause| {
        fail(match cause {
            TryLockError::WouldBlock => CacheSourceError::Busy,
            TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
        })
    })?;
    proof.validate_manager(manager, state.generation)?;
    if !manager.borrowed_storage_complete(&state) {
        return Err(fail(CacheSourceError::PendingStorage));
    }
    proof.validate_pin_counts(&state.lifecycle)?;
    let record = state
        .blocks
        .get(id)
        .ok_or_else(|| fail(CacheSourceError::Identity))?;
    if record.physical.phase() != CacheStoragePhase::Device
        || record
            .disk()
            .and_then(DiskLocation::file_source)
            .is_none_or(|value| !value.same_source(&file))
    {
        return Err(fail(CacheSourceError::Identity));
    }
    let actual = record
        .physical
        .device_resource()
        .ok_or_else(|| fail(CacheSourceError::Identity))?
        .arrays();
    for (actual, expected) in actual.into_iter().zip(arrays) {
        let a = actual
            .try_allocation_info()
            .map_err(|cause| fail(CacheSourceError::ArrayMetadata(cause)))?;
        let b = expected
            .try_allocation_info()
            .map_err(|cause| fail(CacheSourceError::ArrayMetadata(cause)))?;
        if a.is_none() || a != b {
            return Err(fail(CacheSourceError::Identity));
        }
    }
    reporting::update_report_totals_prepared(&mut state)
        .map_err(|cause| proof.error(cause.into()))?;
    let disk_demotions = state
        .telemetry
        .report
        .disk_demotions
        .checked_add(1)
        .ok_or_else(|| fail(CacheSourceError::Overflow))?;
    let layer_demotions = state
        .layer_activity_mut(id.global_layer)
        .disk_demotions
        .checked_add(1)
        .ok_or_else(|| fail(CacheSourceError::Overflow))?;
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
        return Err(proof.error(cause.into()));
    }
    state.telemetry.report.disk_demotions = disk_demotions;
    state.layer_activity_mut(id.global_layer).disk_demotions = layer_demotions;
    drop(state);
    Ok(device)
}

pub(super) fn control_bytes() -> Option<usize> {
    let caller = [
        crate::backend::nn::workspace::OriginalPagedHostEviction::control_bytes()?,
        size_of::<(
            &mut PreparedCacheHostPromotion,
            &OriginalPagedHostReturn<'_, '_>,
        )>(),
        size_of::<(
            &CacheResidencyManager,
            &CacheBlockId,
            &DiskLocation,
            [&Array; 2],
            &mut CachePoolReservation,
            &OriginalPagedHostReturn<'_, '_>,
        )>(),
    ];
    prepared_control_bytes::<crate::backend::nn::workspace::OriginalPagedHostEviction<'_, '_>>()?
        .checked_add(
            caller
                .into_iter()
                .try_fold(size_of_val(&caller), usize::checked_add)?,
        )
}

/// Canonical return transports for the actual selected source proof.
pub(super) fn prepared_control_bytes<P>() -> Option<usize> {
    let frames = [
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
        size_of::<Option<CacheBlockArrays>>(),
        size_of::<(
            &CacheResidencyManager,
            &CacheBlockId,
            &DiskLocation,
            &mut CachePoolReservation,
            &P,
            [&Array; 2],
            u64,
            u64,
        )>(),
        size_of::<Result<CacheBlockArrays, Exception>>(),
        size_of::<CacheBlockArrays>(),
        size_of::<MlxCacheBlockStorage>(),
        size_of::<Option<DiskLocation>>(),
        size_of::<(&DiskLocation, CacheFileSource)>(),
        size_of::<Option<CacheFileSource>>(),
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

//! Calling-thread host publication followed by the existing Device transfer.
use super::*;
use crate::backend::runtime::residency::dense_stream::{BackgroundHostReadService, BackgroundHostServiceError};

pub(crate) struct PreparedBackgroundHostWindow {
    publication: Option<PreparedHostPublication>,
    protection: Option<PreparedHostProtection>,
    prefetch: Vec<OffloadUnitId>,
    host_requested: usize,
    source: ForegroundDiskDescriptors,
    custody: OriginalHostSourceCustody,
    funding: HostMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
enum WindowCause {
    #[error("background source: {0}")]
    Source(#[from] BackgroundHostReadFailure),
    #[error("background worker: {0}")]
    Service(#[from] BackgroundHostServiceError),
    #[error("background host/device acquisition: {0}")]
    Residency(#[from] ResidencyError),
}
/// The consumed window and exact cause retire before its original host custody.
#[derive(thiserror::Error)]
#[error("{cause}")]
pub struct BackgroundHostWindowFailure {
    #[source]
    cause: WindowCause,
    retained: PreparedBackgroundHostWindow,
}
impl std::fmt::Debug for BackgroundHostWindowFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackgroundHostWindowFailure")
            .field("cause", &self.cause)
            .field("prefetch_units", &self.retained.prefetch.len()).finish()
    }
}
impl PreparedBackgroundHostWindow {
    pub(crate) fn control_bytes(
        host: &ForegroundDiskWindowPlan,
        device: &ForegroundDiskWindowPlan,
        group: &str,
        active: &[OffloadUnitId],
    ) -> Option<usize> {
        let fixed = [
            size_of::<Self>(), size_of::<Option<Self>>(),
            size_of::<Vec<OffloadUnitId>>(), size_of::<OffloadUnitId>(),
            Layout::array::<OffloadUnitId>(host.read_units().len()).ok()?.size(),
            size_of::<Result<Self, BackgroundHostReadFailure>>(),
            size_of::<Result<ResidentTransfer, WindowCause>>(),
            size_of::<Result<ResidentTransfer, ResidencyError>>(),
            size_of::<WindowCause>(), size_of::<BackgroundHostWindowFailure>(),
            size_of::<Box<BackgroundHostWindowFailure>>(),
            Layout::new::<BackgroundHostWindowFailure>().size(),
            size_of::<BackgroundSourceAttempt<'_>>(),
            size_of::<Result<BackgroundSourceAttempt<'_>, BackgroundHostServiceError>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<MutexGuard<'_, ManagerState>>(),
            size_of::<Result<MutexGuard<'_, ManagerState>, std::sync::TryLockError<MutexGuard<'_, ManagerState>>>>(),
            size_of::<Result<bool, ResidencyLedgerError>>(),
            size_of::<Result<(), WindowCause>>(),
            size_of::<std::slice::Iter<'_, OffloadUnitId>>(),
            size_of::<OriginalHostPublicationSlots<'_>>(),
            OriginalResidencySlots::control_bytes()?,
            usize::try_from(ResidentTransfer::original_retirement_control_bytes()?).ok()?,
            size_of::<(&ResidencyManager, &ForegroundDiskWindowPlan, &ForegroundDiskWindowPlan, &str, &[OffloadUnitId], usize, &OriginalHostSourceCustody, &HostMetadataFunding)>(),
            size_of::<(&mut Self, &ResidencyManager, &BackgroundHostReadService, &[(OffloadUnitId, u64)], &mut OriginalResidencySlots<'_>, &mut OriginalResidencySlots<'_>, &safemlx::OriginalScopeObserver)>(),
        ];
        let mut bytes = fixed.into_iter().try_fold(size_of_val(&fixed), usize::checked_add)?;
        for id in host.read_units() { bytes = bytes.checked_add(id.as_str().len())?; }
        bytes.checked_add(PreparedHostPublication::host_bytes(device)?)?
            .checked_add(PreparedHostProtection::host_bytes(group, active)?)
    }
    pub(crate) fn prepare(
        manager: &ResidencyManager,
        host: &ForegroundDiskWindowPlan,
        device: &ForegroundDiskWindowPlan,
        group: &str,
        active: &[OffloadUnitId],
        host_requested: usize,
        custody: OriginalHostSourceCustody,
        funding: HostMetadataFunding,
    ) -> Result<Self, BackgroundHostReadFailure> {
        let fail = |cause| BackgroundHostReadFailure::Source { cause, custody: custody.clone(), funding: funding.clone() };
        if !host.matches_manager(manager) || !device.matches_manager(manager)
            || !host.source().same_source(device.source()) {
            return Err(fail(Cause::Identity));
        }
        manager.inner.validate_operation_custody(&custody.metadata_custody()).map_err(|cause| fail(cause.into()))?;
        // Each actual child pays its own constructor below. Reserve only the
        // enclosing final rows/frames here so no child metadata is charged twice.
        let own = Self::control_bytes(host, device, group, active)
            .and_then(|n| n.checked_sub(PreparedHostPublication::host_bytes(device)?))
            .and_then(|n| n.checked_sub(PreparedHostProtection::host_bytes(group, active)?))
            .ok_or_else(|| fail(Cause::Funding(eredu_core::HostMetadataFundingError::Overflow)))?;
        funding.reserve_metadata(own).map_err(|cause| fail(cause.into()))?;
        let publication = PreparedHostPublication::prepare(manager, device, custody.clone(), funding.clone())?;
        let protection = PreparedHostProtection::prepare(manager, group, active, custody.clone(), funding.clone())?;
        let mut prefetch = Vec::new();
        prefetch.try_reserve_exact(host.read_units().len()).map_err(|cause| fail(cause.into()))?;
        for id in host.read_units() { prefetch.push(id.clone()); }
        Ok(Self { publication: Some(publication), protection: Some(protection), prefetch,
            source: host.source().clone(), host_requested, custody, funding })
    }
    fn fail(self, cause: WindowCause) -> ResidencyError {
        ResidencyError::OriginalHostWindow(Box::new(BackgroundHostWindowFailure { cause, retained: self }))
    }
    fn submit(&self, manager: &ResidencyManager, reads: &BackgroundHostReadService) -> Result<(), WindowCause> {
        if !manager.original_foreground_disk_descriptors().is_some_and(|source| self.source.same_source(source)) {
            return Err(ResidencyError::OriginalOperationDomain.into());
        }
        manager.inner.validate_operation_custody(&self.custody.metadata_custody())
            .map_err(ResidencyError::OriginalCache)?;
        for id in &self.prefetch {
            let resident = {
                let state = manager.inner.state.try_lock().map_err(|cause| match cause {
                    std::sync::TryLockError::WouldBlock => ResidencyError::OriginalManagerBusy,
                    std::sync::TryLockError::Poisoned(_) => ResidencyError::StatePoisoned,
                })?;
                if !self.source.units().any(|unit| state.control.unit(id).is_some_and(|current| current == unit)) {
                    return Err(ResidencyError::OriginalOperationDomain.into());
                }
                state.control.ledger().is_resident(id, MemoryTier::Host).map_err(ResidencyError::from)?
            };
            // A host hit reuses its actual source backing. A miss can consume
            // only the next prepared occurrence after the idle transition.
            if !resident { reads.submit(id)?; }
        }
        Ok(())
    }
    /// Consume one exact host window and a separately prepared Host acquisition
    /// before the existing Device attempt. The worker never borrows manager,
    /// device tensors, native observer or a native buffer allowance.
    pub(crate) fn promote(
        mut self,
        manager: &ResidencyManager,
        reads: &BackgroundHostReadService,
        requests: &[(OffloadUnitId, u64)],
        host: &mut OriginalResidencySlots<'_>,
        device: &mut OriginalResidencySlots<'_>,
        observer: &safemlx::OriginalScopeObserver,
    ) -> Result<ResidentTransfer, ResidencyError> {
        let attempt = match reads.attempt() { Ok(value) => value, Err(cause) => return Err(self.fail(cause.into())) };
        let outcome = (|| -> Result<ResidentTransfer, WindowCause> {
            transfer::validate_original_observer(observer)?;
            if host.background_host.is_some() || device.background_host.is_some() {
                return Err(ResidencyError::OriginalOperationDomain.into());
            }
            // Fence earlier callbacks and retire stale ready buffers before
            // this window can submit any new source allocation.
            reads.advance_window(&self.prefetch)?;
            self.protection.take().ok_or(ResidencyError::OriginalOperationDomain)?.apply(manager)?;
            self.submit(manager, reads)?;
            let retained_host = {
                let mut slots = host.reborrow();
                slots.background_host = Some(OriginalHostPublicationSlots { publication: &mut self.publication, reads });
                manager.acquire_many_with_original_transfer(
                    requests.get(..self.host_requested).ok_or(ResidencyError::OriginalOperationDomain)?,
                    MemoryTier::Host, &mut slots, observer)?
            };
            let transfer = manager.acquire_many_with_original_transfer(requests, MemoryTier::Device, device, observer);
            // Host publication is synchronous. Device transfer handlers already
            // retain the exact host owners before their lease pins retire here.
            retained_host.retire_completed_original();
            Ok(transfer?)
        })();
        match outcome {
            Ok(value) => { attempt.succeed(); Ok(value) }
            Err(cause) => Err(self.fail(cause)),
        }
    }
}

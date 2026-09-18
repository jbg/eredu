//! One exact host-protection transition on the calling thread.
use super::*;

/// Owns the actual selected host window and a declared finite group name.
/// Construction grants no read or device operation. Consuming apply performs
/// the same neutral protection transition as ordinary dense scheduling.
pub(crate) struct PreparedHostProtection {
    group: String,
    active: Vec<OffloadUnitId>,
    source: ForegroundDiskDescriptors,
    custody: OriginalHostSourceCustody,
    funding: HostMetadataFunding,
}
impl PreparedHostProtection {
    pub(crate) fn host_bytes(group: &str, active: &[OffloadUnitId]) -> Option<usize> {
        let fixed = [
            group.len(),
            Layout::array::<OffloadUnitId>(active.len()).ok()?.size(),
            size_of::<Self>(),
            size_of::<Result<Self, BackgroundHostReadFailure>>(),
            size_of::<String>(),
            size_of::<Vec<OffloadUnitId>>(),
            size_of::<BackgroundHostReadFailure>(),
            size_of::<Result<(), BackgroundHostReadFailure>>(),
            size_of::<MutexGuard<'_, ManagerState>>(),
            size_of::<std::sync::TryLockError<MutexGuard<'_, ManagerState>>>(),
            size_of::<
                Result<
                    MutexGuard<'_, ManagerState>,
                    std::sync::TryLockError<MutexGuard<'_, ManagerState>>,
                >,
            >(),
            size_of::<Result<(), ResidencyLedgerError>>(),
            size_of::<Result<(), eredu_runtime::working_memory::WorkingMemoryError>>(),
            size_of::<Option<&str>>(),
            size_of::<std::slice::Iter<'_, OffloadUnitId>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, OffloadUnitId>>>(),
            size_of::<Vec<OffloadUnitId>>(),
            size_of::<(
                &ResidencyManager,
                &str,
                &[OffloadUnitId],
                &OriginalHostSourceCustody,
                &HostMetadataFunding,
            )>(),
            size_of::<(
                &mut eredu_runtime::ResidencyController,
                &str,
                &[OffloadUnitId],
                MemoryTier,
            )>(),
        ];
        let bytes = fixed
            .into_iter()
            .try_fold(size_of_val(&fixed), usize::checked_add)?;
        active
            .iter()
            .try_fold(bytes, |sum, id| sum.checked_add(id.as_str().len()))
    }
    pub(crate) fn prepare(
        manager: &ResidencyManager,
        group: &str,
        active: &[OffloadUnitId],
        custody: OriginalHostSourceCustody,
        funding: HostMetadataFunding,
    ) -> Result<Self, BackgroundHostReadFailure> {
        let fail = |cause| BackgroundHostReadFailure::Source {
            cause,
            custody: custody.clone(),
            funding: funding.clone(),
        };
        // Fixed failures precede any ordinary error path that could clone IDs.
        let source = manager
            .original_foreground_disk_descriptors()
            .ok_or_else(|| fail(Cause::Identity))?;
        manager
            .inner
            .validate_operation_custody(&custody.metadata_custody())
            .map_err(|cause| fail(cause.into()))?;
        for (ordinal, id) in active.iter().enumerate() {
            if active[..ordinal].contains(id) || !source.units().any(|unit| unit.id() == id) {
                return Err(fail(Cause::Identity));
            }
        }
        let bytes = Self::host_bytes(group, active).ok_or_else(|| {
            fail(Cause::Funding(
                eredu_core::HostMetadataFundingError::Overflow,
            ))
        })?;
        funding
            .reserve_metadata(bytes)
            .map_err(|cause| fail(cause.into()))?;
        {
            let state = manager.inner.state.try_lock().map_err(|cause| {
                fail(match cause {
                    std::sync::TryLockError::WouldBlock => Cause::Busy,
                    std::sync::TryLockError::Poisoned(_) => Cause::Poisoned,
                })
            })?;
            if state.control.ledger().prepared_group_name(group).is_none() {
                return Err(fail(Cause::Identity));
            }
        }
        let mut selected = Vec::new();
        selected
            .try_reserve_exact(active.len())
            .map_err(|cause| fail(cause.into()))?;
        for id in active {
            selected.push(id.clone());
        }
        let mut name = String::new();
        name.try_reserve_exact(group.len())
            .map_err(|cause| fail(cause.into()))?;
        name.push_str(group);
        Ok(Self {
            group: name,
            active: selected,
            source: source.clone(),
            custody,
            funding,
        })
    }
    /// A host-table mutation only. No ordinary manager lock/reaping, native
    /// completion, transfer, queue admission or payload read is performed here.
    pub(crate) fn apply(self, manager: &ResidencyManager) -> Result<(), BackgroundHostReadFailure> {
        let fail = |cause| BackgroundHostReadFailure::Source {
            cause,
            custody: self.custody.clone(),
            funding: self.funding.clone(),
        };
        if !manager
            .original_foreground_disk_descriptors()
            .is_some_and(|source| source.same_source(&self.source))
            || manager
                .inner
                .failed_transfer
                .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(fail(Cause::Identity));
        }
        manager
            .inner
            .validate_operation_custody(&self.custody.metadata_custody())
            .map_err(|cause| fail(cause.into()))?;
        let mut state = manager.inner.state.try_lock().map_err(|cause| {
            fail(match cause {
                std::sync::TryLockError::WouldBlock => Cause::Busy,
                std::sync::TryLockError::Poisoned(_) => Cause::Poisoned,
            })
        })?;
        if state
            .control
            .ledger()
            .prepared_group_name(&self.group)
            .is_none()
        {
            return Err(fail(Cause::Identity));
        }
        state
            .control
            .protect_group_window(&self.group, &self.active, MemoryTier::Host)
            .map_err(|cause| fail(Cause::Slots(ResidencyError::Ledger(cause))))
    }
}

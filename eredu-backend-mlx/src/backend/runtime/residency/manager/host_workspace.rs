//! Cold facts for the exact per-name Host copy on its selected native stream.
//!
//! The snapshot retains source buffers without residency leases. Post-admission
//! custody explicitly pins the revalidated stores to prevent eviction/fallback.

use std::{
    ops::Range,
    sync::{Arc, atomic::Ordering},
};

use eredu_checkpoint::recipe::{RecipeDtype, RecipeMetadataView};
use eredu_core::residency::{MemoryTier, OffloadUnitId};
use eredu_runtime::{
    residency::{OffloadUnit, WeightBinding},
    working_memory::{OriginalTextControlGuard, OriginalTextMetadataCustody, WorkingMemoryError},
};
use safemlx::{
    AllocationInfo, Dtype, HostTransferMetadataSnapshot, HostTransferPolicy,
    HostTransferStorageKind, ImmutableHostTransferBuffer,
};

use super::{
    ResidencyError, ResidencyManager, ResidentHostOwner, ResidentLeaseStorage, ResidentUnitLease,
    RetainedHostBuffer, transfer::ManagerState,
};
use crate::backend::{
    nn::workspace::NativeAllocationFacts,
    ordinary_retirement::OrdinaryRetirement,
    runtime::checkpoint::recipe::{mlx_dtype, mlx_dtype_if_supported},
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum HostCopyWorkspaceError {
    #[error("host-copy workspace is unavailable: {reason}")]
    Unproved {
        reason: &'static str,
        #[source]
        source: WorkingMemoryError,
    },
    #[error("host-copy residency inspection failed: {0}")]
    Residency(#[source] ResidencyError),
    #[error("host-copy immutable metadata inspection failed: {0}")]
    Metadata(#[source] safemlx::HostTransferMetadataError),
    #[error("host-copy recipe inspection failed: {0}")]
    Recipe(#[from] eredu_checkpoint::recipe::RecipeError),
    #[error("host-copy allocation capacity is unavailable: {0}")]
    Capacity(#[from] eredu_nn::Error),
    #[error("host-copy stream inspection failed: {0}")]
    Native(#[from] safemlx::error::Exception),
    #[error("host-copy scalar stream inspection failed: {0}")]
    StreamCopy(#[from] safemlx::StreamCopyCause),
    #[error("host-copy pin storage failed: {0}")]
    Storage(#[source] WorkingMemoryError),
}

// The error source and a queued failed pin prefix retain independent raw
// aliases. Reclaiming either one cannot refund the other owner's allocation.
#[derive(Debug, thiserror::Error)]
#[error("original initial host pin: {cause}")]
struct InitialHostPinFailure {
    #[source]
    cause: HostCopyWorkspaceError,
    _custody: OriginalTextMetadataCustody,
}

impl HostCopyWorkspaceError {
    fn unknown(reason: &'static str) -> Self {
        Self::Unproved {
            reason,
            source: WorkingMemoryError::UnknownBound,
        }
    }
    fn mismatch(reason: &'static str) -> Self {
        Self::Unproved {
            reason,
            source: WorkingMemoryError::IdentityMismatch,
        }
    }
}

/// One actual copy dispatch, including names declared as aliases.
pub(crate) struct HostCopyBinding {
    // Restores the exact host-map dispatch order after in-place identity checks.
    dispatch_ordinal: usize,
    binding: WeightBinding,
    source: RetainedHostBuffer,
    metadata: HostTransferMetadataSnapshot,
    fresh_capacity_bytes: u64,
    output_capacity_bytes: u64,
}

// Native handle Debug reads metadata through ordinary housekeeping entrypoints.
// A cold snapshot's formatting must remain metadata-only too.
impl std::fmt::Debug for HostCopyBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostCopyBinding")
            .field("binding", &self.binding)
            .field("metadata", &self.metadata)
            .field("fresh_capacity_bytes", &self.fresh_capacity_bytes)
            .field("output_capacity_bytes", &self.output_capacity_bytes)
            .finish()
    }
}

impl HostCopyBinding {
    pub(crate) fn binding(&self) -> &WeightBinding {
        &self.binding
    }
    pub(crate) fn name(&self) -> &str {
        self.binding.name()
    }
    pub(crate) fn source(&self) -> &RetainedHostBuffer {
        &self.source
    }
    pub(crate) fn shape(&self) -> &[i32] {
        self.metadata.shape()
    }
    pub(crate) fn dtype(&self) -> Dtype {
        self.metadata.dtype()
    }
    pub(crate) fn copy_output_layout(&self) -> safemlx::HostTransferArrayLayout {
        self.metadata.copy_output_layout()
    }
    pub(crate) fn logical_bytes(&self) -> u64 {
        self.binding.expected_bytes()
    }
    pub(crate) fn source_allocation(&self) -> AllocationInfo {
        self.metadata.allocation()
    }
    /// Additional destination capacity if this dispatch does not alias its source.
    pub(crate) fn fresh_capacity_bytes(&self) -> u64 {
        self.fresh_capacity_bytes
    }
    /// Complete output backing envelope across fresh allocation and source alias.
    pub(crate) fn output_capacity_bytes(&self) -> u64 {
        self.output_capacity_bytes
    }
}

#[derive(Debug)]
pub(crate) struct HostCopyUnit {
    definition: OffloadUnit,
    request_ordinal: usize,
    copies: Range<usize>,
    fresh_capacity_bytes: u64,
}

impl HostCopyUnit {
    pub(crate) fn id(&self) -> &OffloadUnitId {
        self.definition.id()
    }
    pub(crate) fn definition(&self) -> &OffloadUnit {
        &self.definition
    }
    pub(crate) fn fresh_capacity_bytes(&self) -> u64 {
        self.fresh_capacity_bytes
    }
}

#[derive(Debug)]
pub(crate) struct HostCopyWorkspaceData {
    // The weak owner prevents manager-shell ABA; the actual private stream
    // handle must also remain the one whose device was checked cold.
    manager: super::ManagerWeak,
    stream_handle: usize,
    destination: safemlx::StreamCopyPlan<()>,
    allocation: NativeAllocationFacts,
    units: Vec<HostCopyUnit>,
    copies: Vec<HostCopyBinding>,
    fresh_capacity_bytes: u64,
    unique_source_bytes: u64,
}

// Every strong clone owns the source account independently; no raw Arc/Weak
// escapes, so the final shared allocation retires before its custody.
#[derive(Clone)]
pub(crate) struct HostCopyWorkspace {
    value: Arc<HostCopyWorkspaceData>,
    geometry: Option<HostCopyIdentity>,
    custody: Option<super::ManagerCustody>,
}
impl std::ops::Deref for HostCopyWorkspace {
    type Target = HostCopyWorkspaceData;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
impl std::fmt::Debug for HostCopyWorkspace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.value.fmt(f)
    }
}

mod construction;
mod snapshot;
pub(crate) use snapshot::HostCopyIdentity;

/// Keeps validated source storage at its actual published tier through a quoted run.
/// Destruction only stages the leases; actual unpinning runs at an ordinary
/// retirement boundary, outside native and manager locks.
pub(crate) struct HostCopySourcePins {
    leases: OrdinaryRetirement<HostPinLeases>,
}

// The retirement node is deallocated before this value is dropped. Its Vec and
// every lease/key retire before raw metadata custody; queued cleanup cannot
// refund the charge while those allocations still exist.
struct HostPinLeases {
    values: Vec<ResidentUnitLease>,
    // A qualified resident source may have only a Device ledger row. Its exact
    // immutable host backup remains owned here without creating Host residency.
    _snapshot: Option<HostCopyWorkspace>,
    _custody: Option<super::ResidencyControlCustody>,
}

impl std::fmt::Debug for HostCopySourcePins {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostCopySourcePins")
            .field("units", &self.leases.values.len())
            .finish()
    }
}

impl HostCopyWorkspace {
    pub(crate) fn destination_device_index(&self) -> i32 {
        self.destination.device_index()
    }
    pub(crate) fn destination_device_type(&self) -> safemlx::DeviceType {
        self.destination.device_type()
    }
    pub(crate) fn cloned_definition_payload_bytes(definition: &OffloadUnit) -> Option<usize> {
        super::operation_source::unit_clone_payload_bytes(definition)
    }
    /// Buffers owned by this retained cold snapshot, excluding its inline value
    /// and previously registered source/manager shared owners. Every declaration
    /// is a clone, so the shared clone worker gives its exact requested payload.
    pub(crate) fn retained_payload_bytes(&self) -> Option<usize> {
        use std::alloc::Layout;
        if self.custody.is_some() {
            return Some(0);
        }
        let owner = usize::try_from(
            eredu_runtime::working_memory::OriginalHostMetadataCustody::shared_storage_bytes(
                Layout::new::<HostCopyWorkspaceData>(),
            )
            .ok()?,
        )
        .ok()?;
        let bytes = Layout::array::<HostCopyUnit>(self.units.capacity())
            .ok()?
            .size()
            .checked_add(owner)?
            .checked_add(
                Layout::array::<HostCopyBinding>(self.copies.capacity())
                    .ok()?
                    .size(),
            )?;
        let bytes = self.units.iter().try_fold(bytes, |bytes, unit| {
            bytes.checked_add(super::operation_source::unit_clone_payload_bytes(
                &unit.definition,
            )?)
        })?;
        self.copies.iter().try_fold(bytes, |bytes, copy| {
            bytes
                .checked_add(super::operation_source::binding_clone_payload_bytes(
                    &copy.binding,
                )?)?
                .checked_add(copy.metadata.owned_payload_bytes()?)
        })
    }

    /// The range belongs to a unit borrowed from this exact snapshot. Rows are
    /// retained once in the final destination, in the original dispatch order.
    pub(crate) fn copies(&self, unit: &HostCopyUnit) -> &[HostCopyBinding] {
        &self.copies[unit.copies.clone()]
    }

    pub(crate) fn units(&self) -> &[HostCopyUnit] {
        &self.units
    }
    pub(crate) fn fresh_capacity_bytes(&self) -> u64 {
        self.fresh_capacity_bytes
    }
    pub(crate) fn unique_source_bytes(&self) -> u64 {
        self.unique_source_bytes
    }
    /// Transfer input is copied directly by the selected native worker; there
    /// is no separate numerical Host staging allocation.
    pub(crate) fn host_staging_bytes(&self) -> u64 {
        0
    }

    /// Rechecks definitions and physical sources without acquiring a lease or
    /// repairing a missing store. Both source modes compare the same immutable
    /// owners and exact selected stream under one manager loan; they never
    /// reconstruct a snapshot during validation. Success still requires the executor to retain
    /// its own residency authority across validation and subsequent dispatch.
    pub(crate) fn validate_sources(
        &self,
        manager: &ResidencyManager,
    ) -> Result<(), HostCopyWorkspaceError> {
        let state = manager
            .inner
            .state
            .lock()
            .map_err(|_| HostCopyWorkspaceError::Residency(ResidencyError::StatePoisoned))?;
        self.validate_sources_locked(manager, &state)
    }

    /// Atomically revalidates and pins only already-ready host copies. This
    /// never repairs a missing source, polls work, or records access demand.
    /// Successful custody updates recency through the ledger's existing pin
    /// operation. Rejected snapshot validation leaves all pins unchanged.
    pub(crate) fn pin_sources(
        &self,
        manager: &ResidencyManager,
    ) -> Result<HostCopySourcePins, HostCopyWorkspaceError> {
        self.pin_sources_impl(manager, None)
    }

    /// Original installation consumes the retained immutable-owner decision.
    /// Equivalent-but-replaced wrappers must be requoted; no provider metadata
    /// callback, recipe inference or native shape allocation runs in this check.
    pub(crate) fn pin_retained_sources(
        &self,
        manager: &ResidencyManager,
        controls: &OriginalTextControlGuard,
    ) -> Result<HostCopySourcePins, HostCopyWorkspaceError> {
        self.pin_retained_sources_with_custody(manager, controls.metadata_custody().into())
    }

    pub(crate) fn pin_retained_sources_with_custody(
        &self,
        manager: &ResidencyManager,
        custody: eredu_runtime::working_memory::OriginalOperationMetadataCustody,
    ) -> Result<HostCopySourcePins, HostCopyWorkspaceError> {
        self.custody
            .as_ref()
            .ok_or(HostCopyWorkspaceError::Storage(
                WorkingMemoryError::UnknownBound,
            ))?
            .validate_operation_custody(&custody)
            .map_err(HostCopyWorkspaceError::Storage)?;
        self.retained_pin_control_bytes()
            .ok_or(HostCopyWorkspaceError::Storage(
                WorkingMemoryError::UnknownBound,
            ))?;
        self.pin_sources_impl(manager, Some(custody.into()))
    }

    /// Ordinary request installation pays the same pin destinations and keeps
    /// the actual immutable source at its published tier until completion.
    /// This supplies host custody only, with no original-operation observer.
    pub(crate) fn pin_retained_sources_with_host(
        &self,
        manager: &ResidencyManager,
        funding: &eredu_nn::workspace::HostMetadataFunding,
    ) -> Result<HostCopySourcePins, HostCopyWorkspaceError> {
        let bytes = self
            .retained_pin_control_bytes()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(HostCopyWorkspaceError::Storage(
                WorkingMemoryError::Overflow,
            ))?;
        funding.reserve_metadata(bytes).map_err(|cause| {
            HostCopyWorkspaceError::Storage(WorkingMemoryError::MetadataConstruction(
                eredu_nn::workspace::WorkspaceMetadataError::Funding(cause),
            ))
        })?;
        self.pin_sources_impl(manager, Some(funding.clone().into()))
    }

    /// One initial-quote pin through the accepted quote controls. Mapping a
    /// failure happens after the pin worker has released its manager lock.
    pub(crate) fn pin_initial_sources(
        &self,
        manager: &ResidencyManager,
        controls: &OriginalTextControlGuard,
    ) -> Result<HostCopySourcePins, eredu_core::SharedBackendFailure> {
        self.pin_retained_sources(manager, controls)
            .map_err(|cause| {
                eredu_core::SharedBackendFailure::new(
                    eredu_core::BackendFailureKind::Other,
                    InitialHostPinFailure {
                        cause,
                        _custody: controls.metadata_custody(),
                    },
                )
            })
    }

    /// The same pin producer plus the independent, fixed retained failure
    /// transport used by the initial quote. No source/manager backing is added.
    pub(crate) fn initial_pin_control_bytes(&self) -> Option<u64> {
        use std::mem::size_of;
        let failure = eredu_core::SharedBackendFailure::control_bytes::<InitialHostPinFailure>()?;
        let controls = [
            size_of::<&Self>(),
            size_of::<&ResidencyManager>(),
            size_of::<&OriginalTextControlGuard>(),
            size_of::<Result<HostCopySourcePins, eredu_core::SharedBackendFailure>>(),
            size_of::<InitialHostPinFailure>(),
            size_of::<eredu_core::SharedBackendFailure>(),
            size_of::<OriginalTextMetadataCustody>(),
        ]
        .into_iter()
        .try_fold(failure, usize::checked_add)?;
        self.retained_pin_control_bytes()?
            .checked_add(u64::try_from(controls).ok()?)
    }

    /// Fresh pin destinations only. The existing snapshot, manager/shared host
    /// stores and ordinary retirement TLS are separate retained owners.
    pub(crate) fn retained_pin_control_bytes(&self) -> Option<u64> {
        crate::backend::runtime::residency::storage::native_storage::Bank::shared_borrowed_owner_bytes()?;
        use std::{alloc::Layout, mem::size_of};
        let count = self.units.len();
        let vectors = Layout::array::<OffloadUnitId>(count)
            .ok()?
            .size()
            .checked_add(Layout::array::<ResidentUnitLease>(count).ok()?.size())?
            .checked_add(
                Layout::array::<(MemoryTier, ResidentLeaseStorage)>(count)
                    .ok()?
                    .size(),
            )?;
        let names = self.units.iter().try_fold(0usize, |sum, unit| {
            sum.checked_add(Layout::array::<u8>(unit.id().as_str().len()).ok()?.size())
        })?;
        let controls = [
            size_of::<HostCopySourcePins>(),
            size_of::<HostPinLeases>(),
            size_of::<Vec<OffloadUnitId>>(),
            size_of::<Vec<(MemoryTier, ResidentLeaseStorage)>>(),
            size_of::<(MemoryTier, ResidentLeaseStorage)>(),
            size_of::<Option<HostCopyWorkspace>>(),
            size_of::<MemoryTier>(),
            size_of::<Result<HostCopySourcePins, HostCopyWorkspaceError>>(),
            size_of::<Result<(), std::collections::TryReserveError>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<Option<super::ResidencyControlCustody>>(),
            size_of::<super::ResidencyControlCustody>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<std::sync::MutexGuard<'static, ManagerState>>(),
        ]
        .into_iter()
        .try_fold(
            vectors
                .checked_add(names)?
                .checked_add(self.source_validation_control_bytes()?)?,
            usize::checked_add,
        )?;
        u64::try_from(controls)
            .ok()?
            .checked_add(OrdinaryRetirement::<HostPinLeases>::control_bytes()?)
    }

    fn validate_sources_locked(
        &self,
        manager: &ResidencyManager,
        state: &ManagerState,
    ) -> Result<(), HostCopyWorkspaceError> {
        if !self.manager.ptr_eq(&manager.inner.downgrade())
            || self.stream_handle != state.device_stream.as_ptr().ctx as usize
            || !self.destination.matches_source(&state.device_stream)
        {
            return Err(HostCopyWorkspaceError::mismatch(
                "retained copy stream changed",
            ));
        }
        manager.validate_host_copy_state(&state)?;
        for unit in &self.units {
            let definition = state.control.unit(unit.id()).ok_or_else(|| {
                HostCopyWorkspaceError::mismatch("retained host unit disappeared")
            })?;
            let host = manager
                .host_workspace_source(state, unit.id())
                .ok_or_else(|| {
                    HostCopyWorkspaceError::mismatch("retained host store disappeared")
                })?;
            if definition != unit.definition()
                || host.buffers.len() != unit.copies.len()
                || self.copies(unit).iter().any(|copy| {
                    !host
                        .buffers
                        .get(copy.name())
                        .is_some_and(|source| source.ptr_eq(copy.source()))
                })
            {
                return Err(HostCopyWorkspaceError::mismatch(
                    "retained immutable host owner changed",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn prepared_identity(&self) -> Option<&HostCopyIdentity> {
        self.geometry.as_ref()
    }

    /// Authenticate the persistent source producers without constructing a
    /// snapshot or observing readiness as authority. Request installation still
    /// validates and pins the actual ready stores under the manager lock.
    pub(crate) fn is_original_source_for(
        &self,
        manager: &ResidencyManager,
        windows: &super::OperationWindows,
    ) -> bool {
        self.custody.is_some()
            && self.geometry.is_some()
            && manager.original_checkpoint_source().is_some()
            && manager.owns_original_host_workspace(self, windows)
    }

    pub(in crate::backend::runtime::residency::manager) fn source_validation_control_bytes(
        &self,
    ) -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let controls = [
            size_of::<(
                &Self,
                &ResidencyManager,
                &ManagerState,
                Result<(), HostCopyWorkspaceError>,
            )>(),
            size_of::<(&ResidencyManager, &ManagerState, &OffloadUnitId)>(),
            size_of::<Option<&ResidentHostOwner>>(),
            size_of::<(&Self, &ResidencyManager)>(),
            size_of::<std::sync::MutexGuard<'_, ManagerState>>(),
            size_of::<std::sync::TryLockError<std::sync::MutexGuard<'_, ManagerState>>>(),
            size_of::<Result<(), HostCopyWorkspaceError>>(),
            size_of::<HostCopyWorkspaceError>(),
            size_of::<std::slice::Iter<'_, HostCopyUnit>>(),
            size_of::<std::slice::Iter<'_, HostCopyBinding>>(),
            size_of::<Option<&OffloadUnit>>(),
            size_of::<Option<&super::ResidentHostOwner>>(),
            size_of::<Option<&RetainedHostBuffer>>(),
            size_of::<super::ManagerWeak>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)?
            .checked_add(self.destination.source_comparison_control_bytes()?)
    }
    pub(in crate::backend::runtime::residency::manager) fn same_snapshot(
        &self,
        other: &Self,
    ) -> bool {
        self.custody.is_some() && other.custody.is_some() && Arc::ptr_eq(&self.value, &other.value)
    }
    pub(in crate::backend::runtime::residency::manager) fn validate_original_snapshot(
        &self,
        manager: &ResidencyManager,
    ) -> Result<(), HostCopyWorkspaceError> {
        let state =
            manager.inner.state.try_lock().map_err(|_| {
                HostCopyWorkspaceError::unknown("host source manager is not available")
            })?;
        self.validate_sources_locked(manager, &state)
    }

    fn pin_sources_impl(
        &self,
        manager: &ResidencyManager,
        custody: Option<super::ResidencyControlCustody>,
    ) -> Result<HostCopySourcePins, HostCopyWorkspaceError> {
        // Put custody behind the deferred owner before any fallible prefix.
        let mut leases = OrdinaryRetirement::new(HostPinLeases {
            values: Vec::new(),
            _snapshot: self.custody.as_ref().map(|_| self.clone()),
            _custody: custody,
        });
        let mut ids = Vec::new();
        let mut storage = Vec::new();
        let reserve = |cause| {
            HostCopyWorkspaceError::Storage(WorkingMemoryError::ControlStorageReserve(cause))
        };
        ids.try_reserve_exact(self.units.len()).map_err(reserve)?;
        storage
            .try_reserve_exact(self.units.len())
            .map_err(reserve)?;
        leases
            .values
            .try_reserve_exact(self.units.len())
            .map_err(reserve)?;
        for unit in &self.units {
            ids.push(unit.id().clone());
        }
        let mut state = manager
            .inner
            .state
            .lock()
            .map_err(|_| HostCopyWorkspaceError::Residency(ResidencyError::StatePoisoned))?;
        self.validate_sources_locked(manager, &state)?;
        state
            .control
            .ledger()
            .require_initialized()
            .map_err(|error| HostCopyWorkspaceError::Residency(error.into()))?;
        for id in &ids {
            let host_ready = state
                .control
                .ledger()
                .copy_status(id, MemoryTier::Host)
                .map_err(|error| HostCopyWorkspaceError::Residency(error.into()))?
                .is_some_and(|copy| copy.in_flight().is_none());
            let tier = if host_ready {
                MemoryTier::Host
            } else if self.custody.is_some() && manager.inner.sources.prepared_host(id).is_some() {
                // The immutable source was revalidated above and is retained by
                // the snapshot. Pin the actual ready Device copy, not a fake
                // Host ledger row or an unowned ordinary checkpoint fallback.
                MemoryTier::Device
            } else {
                return Err(HostCopyWorkspaceError::unknown(
                    "host source is not ready for pinning",
                ));
            };
            let copy = state
                .control
                .ledger()
                .copy_status(id, tier)
                .map_err(|error| HostCopyWorkspaceError::Residency(error.into()))?
                .filter(|copy| copy.in_flight().is_none())
                .ok_or_else(|| {
                    HostCopyWorkspaceError::unknown("source storage is not ready for pinning")
                })?;
            copy.pins()
                .checked_add(1)
                .ok_or_else(|| HostCopyWorkspaceError::unknown("source pin count overflow"))?;
            let row = state.storage.get(id).ok_or_else(|| {
                HostCopyWorkspaceError::mismatch("validated source store disappeared")
            })?;
            let retained = match tier {
                MemoryTier::Host => ResidentLeaseStorage::Host(
                    row.host
                        .as_ref()
                        .ok_or_else(|| {
                            HostCopyWorkspaceError::mismatch("validated host store disappeared")
                        })?
                        .clone(),
                ),
                MemoryTier::Device => ResidentLeaseStorage::Device(
                    row.device
                        .as_ref()
                        .ok_or_else(|| {
                            HostCopyWorkspaceError::mismatch("validated device store disappeared")
                        })?
                        .clone(),
                ),
                MemoryTier::Disk => unreachable!("source pins use a published memory tier"),
            };
            storage.push((tier, retained));
        }
        for (index, (id, (tier, _))) in ids.iter().zip(&storage).enumerate() {
            if let Err(error) = state.control.ledger_mut().pin(id, *tier, 0) {
                // No leases exist yet; exact earlier tiers unwind under this
                // same manager loan without recursive Drop/lock acquisition.
                for (pinned, (tier, _)) in ids[..index].iter().zip(&storage[..index]) {
                    state.control.ledger_mut().unpin(pinned, *tier);
                }
                return Err(HostCopyWorkspaceError::Residency(error.into()));
            }
        }
        // All fallible validation and allocations precede the first pin.
        for (id, (tier, storage)) in ids.into_iter().zip(storage) {
            leases.values.push(ResidentUnitLease::with_owner(
                id,
                tier,
                storage,
                manager.inner.downgrade(),
            ));
        }
        drop(state);
        Ok(HostCopySourcePins { leases })
    }
}

fn add(left: u64, right: u64) -> Result<u64, HostCopyWorkspaceError> {
    left.checked_add(right)
        .ok_or_else(|| HostCopyWorkspaceError::unknown("capacity sum overflow"))
}

impl ResidencyManager {
    /// Snapshots the exact initialized host bindings used by the native copy
    /// loop. No completion polling, recovery, payload reads, eviction, array
    /// creation, or fallback materialization occurs. Sources must be immutable
    /// Transfer buffers supported by the selected CPU or Metal copy worker.
    ///
    /// Alias names remain separate destination opportunities because
    /// `prepare_copy_to_device` calls `copy_to_array` for every stored name.
    /// Sources are deduplicated only for their already-live physical capacity.
    pub(crate) fn host_copy_workspace(
        &self,
        ids: &[OffloadUnitId],
        allocation: NativeAllocationFacts,
    ) -> Result<HostCopyWorkspace, HostCopyWorkspaceError> {
        self.host_copy_workspace_impl(ids, allocation, false)
    }

    /// Borrows only the snapshot created by the actual original source loader.
    /// It never constructs a legacy snapshot when that source is unavailable.
    pub(crate) fn prepared_host_copy_workspace(
        &self,
        ids: &[OffloadUnitId],
        allocation: NativeAllocationFacts,
    ) -> Result<HostCopyWorkspace, HostCopyWorkspaceError> {
        self.host_copy_workspace_impl(ids, allocation, true)
    }
    fn host_copy_workspace_impl(
        &self,
        ids: &[OffloadUnitId],
        allocation: NativeAllocationFacts,
        prepared_only: bool,
    ) -> Result<HostCopyWorkspace, HostCopyWorkspaceError> {
        // self.lock() reaps recovery; inspection must not establish readiness.
        let state = self
            .inner
            .state
            .lock()
            .map_err(|_| HostCopyWorkspaceError::Residency(ResidencyError::StatePoisoned))?;
        if let Some(cached) = self.inner.host_workspace.get() {
            if cached.allocation.page_size() != allocation.page_size()
                || cached.units.len() != ids.len()
                || cached
                    .units
                    .iter()
                    .zip(ids)
                    .any(|(unit, id)| unit.id() != id)
            {
                return Err(HostCopyWorkspaceError::mismatch(
                    "prepared host snapshot selection changed",
                ));
            }
            cached.validate_sources_locked(self, &state)?;
            return Ok(cached.clone());
        }
        if prepared_only || self.original_source_custody().is_some() {
            return Err(HostCopyWorkspaceError::unknown(
                "original host snapshot was not initialized",
            ));
        }
        self.host_copy_workspace_locked(&state, ids, allocation)
    }

    fn validate_host_copy_state(&self, state: &ManagerState) -> Result<(), HostCopyWorkspaceError> {
        if self.inner.failed_transfer.load(Ordering::Acquire) {
            return Err(HostCopyWorkspaceError::unknown(
                "manager has failed transfer ownership",
            ));
        }
        for unit in state.control.units() {
            for tier in [MemoryTier::Host, MemoryTier::Device] {
                let status = state
                    .control
                    .ledger()
                    .copy_status(unit.id(), tier)
                    .map_err(|error| HostCopyWorkspaceError::Residency(error.into()))?;
                if status.is_some_and(|copy| copy.in_flight().is_some()) {
                    return Err(HostCopyWorkspaceError::unknown(
                        "manager transfer remains in flight",
                    ));
                }
            }
        }
        Ok(())
    }

    /// The immutable original source owns its paid host backing independently
    /// of its current ledger tier. Resident units may publish only Device rows.
    /// This loan describes that backing; it neither installs a Host row nor
    /// substitutes for the acquisition/pinning checks on an actual transfer.
    fn host_workspace_source<'a>(
        &'a self,
        state: &'a ManagerState,
        id: &OffloadUnitId,
    ) -> Option<&'a ResidentHostOwner> {
        state
            .storage
            .get(id)
            .and_then(|row| row.host.as_ref())
            .or_else(|| self.inner.sources.prepared_host(id))
    }

    fn host_copy_workspace_locked(
        &self,
        state: &ManagerState,
        ids: &[OffloadUnitId],
        allocation: NativeAllocationFacts,
    ) -> Result<HostCopyWorkspace, HostCopyWorkspaceError> {
        let original = self.original_source_custody().is_some();
        self.validate_host_copy_state(state)?;
        let destination = safemlx::StreamCopyPlan::<()>::capture(&state.device_stream)
            .map_err(HostCopyWorkspaceError::StreamCopy)?;
        // Both vectors are final snapshot storage. Their exact populations
        // come from the locked selected stores; no tree nodes or separate
        // duplicate/source index are allocated during construction.
        let reserve = |cause| {
            HostCopyWorkspaceError::Storage(WorkingMemoryError::ControlStorageReserve(cause))
        };
        let mut units = Vec::<HostCopyUnit>::new();
        units.try_reserve_exact(ids.len()).map_err(reserve)?;
        let mut copy_count = 0usize;
        for (request_ordinal, id) in ids.iter().enumerate() {
            let definition = state
                .control
                .unit(id)
                .ok_or_else(|| HostCopyWorkspaceError::mismatch("unknown requested unit"))?;
            let host = self.host_workspace_source(state, id).ok_or_else(|| {
                HostCopyWorkspaceError::unknown("requested unit has no initialized host store")
            })?;
            if host.buffers.len() != definition.bindings().len() {
                return Err(HostCopyWorkspaceError::mismatch(
                    "host binding names differ from unit definition",
                ));
            }
            copy_count = copy_count.checked_add(host.buffers.len()).ok_or(
                HostCopyWorkspaceError::Storage(WorkingMemoryError::Overflow),
            )?;
            units.push(HostCopyUnit {
                definition: definition.clone(),
                request_ordinal,
                copies: 0..0,
                fresh_capacity_bytes: 0,
            });
        }
        // Unstable sorting is in-place and uses no heap workspace. Ordinals
        // restore even empty units exactly, independently of their names.
        units.sort_unstable_by(|a, b| a.id().cmp(b.id()));
        if units.windows(2).any(|pair| pair[0].id() == pair[1].id()) {
            return Err(HostCopyWorkspaceError::mismatch("duplicate requested unit"));
        }
        units.sort_unstable_by_key(|unit| unit.request_ordinal);
        let mut copies = Vec::<HostCopyBinding>::new();
        copies.try_reserve_exact(copy_count).map_err(reserve)?;
        let mut fresh_capacity_bytes = 0;
        for unit in &mut units {
            let id = unit.id();
            let definition = state.control.unit(id).expect("validated locked unit");
            let host = self
                .host_workspace_source(state, id)
                .expect("validated locked host store");
            let start = copies.len();
            let mut unit_capacity = 0;
            // Match dispatch iteration order, not semantic owner order.
            for (name, source) in &host.buffers {
                let binding = definition
                    .bindings()
                    .iter()
                    .find(|binding| binding.name() == name)
                    .ok_or_else(|| {
                        HostCopyWorkspaceError::mismatch("host binding name is undeclared")
                    })?;
                let metadata = if original {
                    source
                        .prepared_metadata()
                        .ok_or_else(|| {
                            HostCopyWorkspaceError::unknown(
                                "original host source lacks retained metadata",
                            )
                        })?
                        .clone()
                } else {
                    source
                        .try_metadata_snapshot()
                        .map_err(|error| match error {
                            safemlx::HostTransferMetadataError::RuntimeBusy => {
                                HostCopyWorkspaceError::unknown("native metadata lock is busy")
                            }
                            error => HostCopyWorkspaceError::Metadata(error),
                        })?
                };
                if metadata.policy() != HostTransferPolicy::Transfer
                    || !(metadata.storage_kind() == HostTransferStorageKind::MetalShared
                        || (destination.device_type() == safemlx::DeviceType::Cpu
                            && metadata.storage_kind() == HostTransferStorageKind::Cpu))
                {
                    return Err(HostCopyWorkspaceError::unknown(
                        "host transfer storage is incompatible with the destination device",
                    ));
                }
                if !copy_shape_is_supported(destination.device_type(), metadata.shape()) {
                    return Err(HostCopyWorkspaceError::unknown(
                        "Host copy shape is outside the selected native worker",
                    ));
                }
                let (owner_id, owner) = if binding.is_alias() {
                    state.control.binding_owner(id, binding).ok_or_else(|| {
                        HostCopyWorkspaceError::mismatch("alias has no canonical source recipe")
                    })?
                } else {
                    (id, binding)
                };
                let inspect = |recipe: RecipeMetadataView<'_>| {
                    let dtype = if original {
                        mlx_dtype_if_supported(recipe.dtype())
                    } else {
                        mlx_dtype(recipe.dtype()).ok()
                    }
                    .ok_or_else(|| {
                        HostCopyWorkspaceError::unknown(
                            "recipe dtype has no native host representation",
                        )
                    })?;
                    let rank = recipe.shape().len();
                    let packed = recipe.dtype() == &RecipeDtype::F4;
                    if packed
                        && (rank == 0 || recipe.shape().last().is_none_or(|last| last % 2 != 0))
                    {
                        return Err(HostCopyWorkspaceError::unknown(
                            "packed recipe has no proved native shape",
                        ));
                    }
                    // The actual encoded F4 lowering halves its final axis.
                    // Check every conversion, even after an earlier mismatch,
                    // preserving the old shape-overflow refusal precedence.
                    let mut matches = metadata.shape().len() == rank;
                    for (axis, mut dimension) in recipe.shape().enumerate() {
                        if packed && axis + 1 == rank {
                            dimension /= 2;
                        }
                        let dimension = i32::try_from(dimension).map_err(|_| {
                            HostCopyWorkspaceError::unknown("native shape dimension overflow")
                        })?;
                        matches &= metadata.shape().get(axis) == Some(&dimension);
                    }
                    let bytes = u64::try_from(metadata.nbytes()).map_err(|_| {
                        HostCopyWorkspaceError::unknown("logical byte count overflow")
                    })?;
                    let source_bytes = u64::try_from(metadata.allocation().bytes())
                        .map_err(|_| HostCopyWorkspaceError::unknown("source capacity overflow"))?;
                    if !matches
                        || metadata.dtype() != dtype
                        || bytes != binding.expected_bytes()
                        || bytes != recipe.byte_len()
                        || source_bytes < bytes
                    {
                        return Err(HostCopyWorkspaceError::mismatch(
                            "host geometry or capacity differs from its binding",
                        ));
                    }
                    Ok((bytes, source_bytes))
                };
                let (bytes, source_bytes) = if original {
                    let recipe = self
                        .original_host_recipe(owner_id, owner.name())
                        .ok_or_else(|| {
                            HostCopyWorkspaceError::mismatch("original host recipe is missing")
                        })?;
                    inspect(recipe.borrowed())?
                } else {
                    owner.with_source_metadata(self.inner.sources.source(owner_id), inspect)??
                };
                let fresh = if original {
                    allocation.fixed_buffer_capacity(bytes).map_err(|_| {
                        HostCopyWorkspaceError::Storage(WorkingMemoryError::Overflow)
                    })?
                } else {
                    allocation.buffer_capacity(bytes)?
                };
                unit_capacity = add(unit_capacity, fresh)?;
                copies.push(HostCopyBinding {
                    dispatch_ordinal: copies.len(),
                    binding: binding.clone(),
                    source: source.clone(),
                    metadata,
                    fresh_capacity_bytes: fresh,
                    output_capacity_bytes: fresh.max(source_bytes),
                });
            }
            fresh_capacity_bytes = add(fresh_capacity_bytes, unit_capacity)?;
            unit.copies = start..copies.len();
            unit.fresh_capacity_bytes = unit_capacity;
        }
        debug_assert_eq!(copies.len(), copy_count);
        copies.sort_unstable_by_key(|copy| copy.source_allocation().identity());
        let mut previous = None;
        let mut unique_source_bytes = 0;
        for copy in &copies {
            let info = copy.source_allocation();
            let bytes = u64::try_from(info.bytes())
                .map_err(|_| HostCopyWorkspaceError::unknown("source capacity overflow"))?;
            match previous {
                Some((identity, prior)) if identity == info.identity() => {
                    if prior != bytes {
                        return Err(HostCopyWorkspaceError::mismatch(
                            "one source identity has conflicting capacities",
                        ));
                    }
                }
                _ => unique_source_bytes = add(unique_source_bytes, bytes)?,
            }
            previous = Some((info.identity(), bytes));
        }
        copies.sort_unstable_by_key(|copy| copy.dispatch_ordinal);
        Ok(HostCopyWorkspace {
            value: Arc::new(HostCopyWorkspaceData {
                manager: self.inner.downgrade(),
                stream_handle: state.device_stream.as_ptr().ctx as usize,
                destination,
                allocation,
                units,
                copies,
                fresh_capacity_bytes,
                unique_source_bytes,
            }),
            geometry: None,
            custody: None,
        })
    }
}

/// The CPU General-copy worker uses positive signed-index geometry. Snapshot
/// shapes are canonical dense Host arrays; rank/dtype dispatch is separately
/// authenticated by the selected native copy-layout query.
pub(super) fn copy_shape_is_supported(device: safemlx::DeviceType, shape: &[i32]) -> bool {
    device != safemlx::DeviceType::Cpu
        || shape
            .iter()
            .try_fold(1_i32, |n, &d| (d > 0).then(|| n.checked_mul(d)).flatten())
            .is_some()
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;

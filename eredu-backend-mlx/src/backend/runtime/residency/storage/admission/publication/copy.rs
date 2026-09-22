//! Completed snapshot destinations through the shared canonical B publisher.
use super::*;
use crate::backend::array_copy::PreparedSavedHostCopy;
use eredu_core::{BackendFailure, HostPreparationAuthority};
use eredu_runtime::working_memory::{
    PreparedWorkspaceCopyPublication, WorkspaceCopyCustody, WorkspaceCopyPublicationPlan,
};
use safemlx::ImmutableHostTransferBuffer;
mod host;
use safemlx::{OriginalBufferBudget, PreparedAllocationOwner};
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
    sync::Arc,
};

type Registration = WorkingMemoryStorage<StorageIdentity>;

pub(crate) struct CopyPublicationLayout {
    registry: WorkspaceCopyPublicationPlan<StorageIdentity>,
    rows: usize,
    host_rows: usize,
    bytes: usize,
}
impl CopyPublicationLayout {
    pub(crate) fn new(native: usize, metadata: usize) -> Result<Self, WorkingMemoryError> {
        Self::with_host(native, 0, metadata)
    }
    pub(crate) fn with_host(
        native: usize,
        host: usize,
        metadata: usize,
    ) -> Result<Self, WorkingMemoryError> {
        let rows = native
            .checked_add(host)
            .ok_or(WorkingMemoryError::Overflow)?
            .checked_add(metadata)
            .ok_or(WorkingMemoryError::Overflow)?;
        let nested = eredu_runtime::HostMetadataKey::maximum_clone_storage_bytes()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let registry = WorkspaceCopyPublicationPlan::new(rows, nested)?;
        let attachment = PreparedAllocationOwner::<Registration>::layout();
        let attachment_frames = [
            attachment
                .allocation_bytes()
                .ok_or(WorkingMemoryError::Overflow)?,
            attachment.preparation_control_bytes(),
            attachment.prepared_bytes(),
            attachment.preparation_failure_bytes(),
            attachment.attachment_failure_bytes(),
            attachment.original_attachment_control_bytes(),
        ];
        let array_attachment = attachment_frames
            .into_iter()
            .try_fold(size_of_val(&attachment_frames), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        let metadata_attachment = eredu_runtime::HostSlotMetadata::copy_attachment_control_bytes()
            .and_then(|n| n.checked_add(size_of::<Registration>())) // actual Box payload
            .and_then(|n| n.checked_add(size_of::<Box<dyn Send + Sync>>()))
            .ok_or(WorkingMemoryError::Overflow)?;
        let frames = [
            registry.requested_bytes(),
            host::control_bytes().ok_or(WorkingMemoryError::UnknownBound)?,
            Layout::array::<Arc<ImmutableHostTransferBuffer>>(host)
                .map_err(|_| WorkingMemoryError::Overflow)?
                .size(),
            safemlx::HostTransferDescriptor::<4>::control_bytes()
                .ok_or(WorkingMemoryError::UnknownBound)?,
            size_of::<
                Result<
                    (),
                    safemlx::PreparedAllocationOwnerError<PreparedAllocationOwner<Registration>>,
                >,
            >(),
            size_of::<(&PreparedSavedHostCopy, &OriginalBufferBudget)>(),
            size_of::<Option<&Arc<ImmutableHostTransferBuffer>>>(),
            size_of::<Result<(), Error>>(),
            size_of::<std::slice::Iter<'_, Arc<ImmutableHostTransferBuffer>>>(),
            Layout::array::<(StorageIdentity, u64, Arc<eredu_core::MemoryPlacement>)>(rows)
                .map_err(|_| WorkingMemoryError::Overflow)?
                .size(),
            usize::try_from(nested)
                .map_err(|_| WorkingMemoryError::Overflow)?
                .checked_mul(rows)
                .ok_or(WorkingMemoryError::Overflow)?,
            array_attachment
                .checked_mul(
                    native
                        .checked_add(host)
                        .ok_or(WorkingMemoryError::Overflow)?,
                )
                .ok_or(WorkingMemoryError::Overflow)?,
            metadata_attachment
                .checked_mul(metadata)
                .ok_or(WorkingMemoryError::Overflow)?,
            usize::try_from(
                super::publication_control_bytes(0).ok_or(WorkingMemoryError::Overflow)?,
            )
            .map_err(|_| WorkingMemoryError::Overflow)?,
            size_of::<Self>(),
            size_of::<PendingCopyPublication>(),
            size_of::<Result<Self, WorkingMemoryError>>(),
            size_of::<Result<PendingCopyPublication, Error>>(),
            size_of::<Result<RetainedStoragePublication, Error>>(),
            size_of::<Option<PreparedAllocationOwner<Registration>>>(),
            size_of::<
                Result<(), safemlx::OriginalBufferError<PreparedAllocationOwner<Registration>>>,
            >(),
            size_of::<Option<Registration>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<HostPreparationAuthority>(),
            // Only one inner cause escapes each failed attempt, but it overlaps
            // the retained outer source. These are actual Box payloads and the
            // transport shared by the selected native/slot attachment workers.
            size_of::<safemlx::OriginalBufferCause>()
                .max(size_of::<safemlx::PreparedAllocationOwnerCause>())
                .max(size_of::<
                    eredu_runtime::HostSlotAttachmentError<WorkingMemoryError>,
                >())
                .max(size_of::<WorkingMemoryError>()),
            size_of::<Box<dyn std::error::Error + Send + Sync>>(),
            BackendFailure::source_retention_peak_bytes::<Failure>()
                .ok_or(WorkingMemoryError::Overflow)?,
        ];
        let bytes = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            registry,
            rows,
            host_rows: host,
            bytes,
        })
    }
    pub(crate) fn control_bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn construct(
        self,
        scope: &WorkingMemoryFundingScope,
        copy: &WorkspaceCopyCustody,
        host: &HostPreparationAuthority,
        budget: Option<&OriginalBufferBudget>,
    ) -> Result<PendingCopyPublication, Error> {
        let attempt = self
            .registry
            .prepare(copy, scope)
            .map_err(Error::PrefillControl)?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(self.rows)
            .map_err(|e| Error::PrefillControl(WorkingMemoryError::ControlStorageReserve(e)))?;
        if entries.capacity() != self.rows {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        if self.host_rows != 0 && budget.is_none() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let mut host_sources = Vec::new();
        host_sources
            .try_reserve_exact(self.host_rows)
            .map_err(|e| Error::PrefillControl(WorkingMemoryError::ControlStorageReserve(e)))?;
        if host_sources.capacity() != self.host_rows {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        Ok(PendingCopyPublication {
            inventory: None,
            entries,
            host_sources,
            attempt,
            failed_attachment: None,
            budget: budget.cloned(),
            started: false,
            _host: host.clone(),
        })
    }
}

pub(crate) struct PendingCopyPublication {
    inventory: Option<RetainedStorage>,
    entries: Vec<(StorageIdentity, u64, Arc<eredu_core::MemoryPlacement>)>,
    host_sources: Vec<Arc<ImmutableHostTransferBuffer>>,
    attempt: PreparedWorkspaceCopyPublication<StorageIdentity>,
    failed_attachment: Option<PreparedAllocationOwner<Registration>>,
    budget: Option<OriginalBufferBudget>,
    started: bool,
    _host: HostPreparationAuthority,
}
impl PendingCopyPublication {
    /// Bind only a positively completed output of this same original copy.
    /// The finite list is paid at construction; no Arc or witness is accepted
    /// from an external caller, and ordinary Host buffers cannot enter it.
    pub(crate) fn bind_host_copy(&mut self, copy: &PreparedSavedHostCopy) -> Result<(), Error> {
        let result = (|| {
            if self.started || self.host_sources.len() == self.host_sources.capacity() {
                return Err(Error::PrefillControl(
                    WorkingMemoryError::PreparationAlreadyStarted,
                ));
            }
            let budget = self
                .budget
                .as_ref()
                .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
            let source = copy
                .completed_for(budget)
                .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
            if self.host_sources.iter().any(|old| Arc::ptr_eq(old, source)) {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            self.host_sources.push(Arc::clone(source));
            Ok(())
        })();
        result.map_err(|cause| retain_failure(cause, &self._host))
    }
    pub(crate) fn publish(
        &mut self,
        inventory: RetainedStorage,
        scope: &WorkingMemoryFundingScope,
    ) -> Result<RetainedStoragePublication, Error> {
        if self.started {
            return Err(Error::PrefillControl(
                WorkingMemoryError::PreparationAlreadyStarted,
            ));
        }
        self.started = true;
        self.inventory = Some(inventory);
        let result = self.publish_inner(scope);
        result.map_err(|cause| retain_failure(cause, &self._host))
    }
    fn publish_inner(
        &mut self,
        scope: &WorkingMemoryFundingScope,
    ) -> Result<RetainedStoragePublication, Error> {
        let inventory = self
            .inventory
            .as_mut()
            .expect("one actual publication inventory");
        for host in &self.host_sources {
            inventory.include_host(Arc::clone(host))?;
        }
        let inventory = &*inventory;
        inventory
            .validate_snapshot_publication(scope.pool())
            .map_err(Error::PrefillControl)?;
        if inventory.validate_inventory_completeness().is_err()
            || inventory.metadata_entries().next().is_some()
            || inventory.capture_entries().next().is_some()
            || inventory.source_capacities().next().is_some()
            || !inventory.byte_buffers.is_empty()
        {
            return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound));
        }
        for (identity, (bytes, array)) in inventory.array_entries() {
            let budget = self
                .budget
                .as_ref()
                .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
            let witness = host::observe(array, budget, &self.host_sources)?;
            if witness.allocation().identity() != *identity
                || witness.allocation().bytes() as u64 != *bytes
            {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            push(
                &mut self.entries,
                StorageIdentity::Native(*identity),
                *bytes,
                placement_handle(&witness.allocation(), scope.pool())?,
            )?;
        }
        if inventory.host_entries().count() != self.host_sources.len() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        for (identity, (bytes, buffer)) in inventory.host_entries() {
            // Pointer equality is only to select the closed immutable owner;
            // the actual descriptor supplies the native generation/capacity.
            if !self
                .host_sources
                .iter()
                .any(|source| std::ptr::eq(buffer.as_ref(), source.as_ref()))
            {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            let descriptor = buffer
                .try_fixed_descriptor::<4>()
                .map_err(|cause| Error::Other(Box::new(cause)))?;
            if descriptor.allocation().identity() != *identity
                || descriptor.allocation().bytes() as u64 != *bytes
            {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            push(
                &mut self.entries,
                StorageIdentity::Native(*identity),
                *bytes,
                placement_handle(&descriptor.allocation(), scope.pool())?,
            )?;
        }
        for (identity, (bytes, token)) in inventory.slot_entries() {
            token
                .prepare_copy_attachment(scope.pool().shared_storage_accounting_id())
                .map_err(Error::PrefillControl)?;
            push(
                &mut self.entries,
                StorageIdentity::HostMetadata(identity.registry_key().clone()),
                *bytes,
                scope.pool().host_placement_handle(),
            )?;
        }
        for (key, bytes, placement) in &self.entries {
            self.attempt
                .push(key.clone(), *bytes, Arc::clone(placement))
                .map_err(Error::PrefillControl)?;
        }
        self.attempt.publish(scope).map_err(Error::PrefillControl)?;
        for (identity, (bytes, array)) in inventory.array_entries() {
            let index = self
                .entries
                .iter()
                .position(|(key, _, _)| key == &StorageIdentity::Native(*identity))
                .expect("prepared native row");
            let registration = self
                .attempt
                .input(index)
                .expect("published unique row")
                .clone();
            let prepared = PreparedAllocationOwner::try_new(registration).map_err(|failure| {
                let (cause, _registration) = failure.into_parts();
                Error::Other(Box::new(cause))
            })?;
            let witness = host::observe(
                array,
                self.budget.as_ref().expect("validated native budget"),
                &self.host_sources,
            )?;
            if witness.allocation().identity() != *identity
                || witness.allocation().bytes() as u64 != *bytes
            {
                self.failed_attachment = Some(prepared);
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            if let Err(failure) = witness.try_attach(prepared) {
                let (cause, prepared) = failure.into_parts();
                self.failed_attachment = Some(prepared);
                return Err(buffer_error(cause));
            }
        }
        for (identity, (_, buffer)) in inventory.host_entries() {
            let index = self
                .entries
                .iter()
                .position(|(key, _, _)| key == &StorageIdentity::Native(*identity))
                .expect("prepared completed Host row");
            let registration = self
                .attempt
                .input(index)
                .expect("published unique Host row")
                .clone();
            let prepared = PreparedAllocationOwner::try_new(registration).map_err(|failure| {
                let (cause, _registration) = failure.into_parts();
                Error::Other(Box::new(cause))
            })?;
            if let Err(failure) = buffer.try_attach_prepared_allocation_owner(prepared) {
                let (cause, prepared) = failure.into_parts();
                self.failed_attachment = Some(prepared);
                return Err(Error::Other(Box::new(cause)));
            }
        }
        let inventory = self.inventory.as_mut().expect("retained prefix");
        inventory.attach_prepared(
            |key| {
                self.entries
                    .iter()
                    .position(|(entry, _, _)| entry == key)
                    .and_then(|index| self.attempt.input(index).cloned())
            },
            &UnquotedOriginalSlotSources::default(),
            None,
            scope.pool().shared_storage_accounting_id(),
            false,
            true,
        )?;
        // Every row now has its physical/table attachment. Drop only the
        // transaction's duplicate registrations before certifying the scope.
        for index in 0..self.entries.len() {
            drop(self.attempt.take_input(index));
        }
        let inventory = self.inventory.take().expect("successful inventory");
        Ok(inventory.finish_publication_with_custody(
            Vec::new,
            UnquotedOriginalSlotSources::default(),
            None,
            Some(self._host.clone()),
            scope.pool().shared_storage_accounting_id(),
        ))
    }
}
fn push(
    entries: &mut Vec<(StorageIdentity, u64, Arc<eredu_core::MemoryPlacement>)>,
    key: StorageIdentity,
    bytes: u64,
    placement: Arc<eredu_core::MemoryPlacement>,
) -> Result<(), Error> {
    if entries.len() == entries.capacity() {
        return Err(Error::PrefillControl(
            WorkingMemoryError::CollectorCapacity {
                kind: eredu_runtime::working_memory::CollectorCapacityKind::InventoryRows,
                used: entries.len(),
                capacity: entries.capacity(),
            },
        ));
    }
    entries.push((key, bytes, placement));
    Ok(())
}
fn buffer_error(cause: safemlx::OriginalBufferCause) -> Error {
    Error::Other(Box::new(cause))
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Error,
    _host: HostPreparationAuthority,
}
pub(crate) fn retain_failure(cause: Error, host: &HostPreparationAuthority) -> Error {
    Error::StorageSource(BackendFailure::from_error(Failure {
        cause,
        _host: host.clone(),
    }))
}

fn placement_handle(
    info: &safemlx::AllocationInfo,
    pool: &eredu_runtime::working_memory::MemoryLedger,
) -> Result<Arc<eredu_core::MemoryPlacement>, Error> {
    crate::backend::managed_memory::allocation_placement_handle(info, pool)
        .map_err(|error| Error::Other(Box::new(error)))
}

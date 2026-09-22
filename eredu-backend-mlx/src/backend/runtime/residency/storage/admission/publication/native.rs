//! Retain the actual mixed inventory through checked native publication.
use super::*;
use crate::backend::runtime::residency::storage::native_storage::{
    retained_failure_at, MlxNativeStorage, NativeStorageRoot,
};
use eredu_runtime::working_memory::{OriginalNativePublication, OriginalTextControlGuard};

pub(crate) struct PendingNativePublication {
    // Physical/source owners precede duplicate keys and every unattached charge.
    // Work's recovery retains this entire value after any terminal refusal.
    inventory: Option<RetainedStorage>,
    entries: Vec<(
        StorageIdentity,
        eredu_runtime::working_memory::StorageAllocation,
    )>,
    original: Option<UnquotedOriginalSlotSources>,
    attempt: Option<OriginalNativePublication<MlxNativeStorage>>,
}
impl PendingNativePublication {
    /// Source-entry destination, successful publication owners and all named
    /// transaction transports. The neutral bank prices its own source outputs.
    pub(crate) fn control_bytes(rows: usize) -> Option<u64> {
        use std::{alloc::Layout, mem::size_of};
        let keys = eredu_runtime::HostMetadataKey::maximum_clone_storage_bytes()?;
        let entries = Layout::array::<(
            StorageIdentity,
            eredu_runtime::working_memory::StorageAllocation,
        )>(rows)
        .ok()?
        .size();
        // Infer the actual borrowed iterator representation without constructing
        // an inventory, cloning a source, or invoking the iterator factory.
        fn iterator_bytes<T>(_: impl FnOnce(&'static RetainedStorage) -> T) -> usize {
            size_of::<T>()
        }
        let fixed = [
            entries,
            Layout::array::<(StorageIdentity, u64)>(rows).ok()?.size(),
            iterator_bytes(borrowed_native_roots),
            size_of::<NativeStorageRoot<'static>>(),
            size_of::<&RetainedStorage>(),
            size_of::<Self>(),
            size_of::<Result<RetainedStoragePublication, Error>>(),
            size_of::<(
                Vec<(
                    StorageIdentity,
                    eredu_runtime::working_memory::StorageAllocation,
                )>,
                UnquotedOriginalSlotSources,
            )>(),
            size_of::<
                Result<
                    (
                        Vec<(
                            StorageIdentity,
                            eredu_runtime::working_memory::StorageAllocation,
                        )>,
                        UnquotedOriginalSlotSources,
                    ),
                    Error,
                >,
            >(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<StorageIdentity>(),
            size_of::<UnquotedOriginalSlotSources>(),
            size_of::<WorkingMemoryError>(),
            size_of::<Option<&eredu_runtime::input::OriginalPreparedWorkspaceSource>>(),
            size_of::<Result<(), Error>>(),
        ];
        u64::try_from(
            fixed
                .into_iter()
                .try_fold(std::mem::size_of_val(&fixed), usize::checked_add)?,
        )
        .ok()?
        .checked_add(keys.checked_mul(u64::try_from(rows).ok()?)?)?
        .checked_add(super::publication_control_bytes(rows)?)
    }

    pub(crate) fn new(
        inventory: RetainedStorage,
        attempt: OriginalNativePublication<MlxNativeStorage>,
    ) -> Self {
        Self {
            inventory: Some(inventory),
            entries: Vec::new(),
            original: None,
            attempt: Some(attempt),
        }
    }

    pub(crate) fn publish(
        &mut self,
        scope: &mut WorkingMemoryFundingScope,
        controls: &OriginalTextControlGuard,
        fixed: Option<&eredu_runtime::working_memory::OriginalResidentResetSource>,
        prepared: Option<&eredu_runtime::input::OriginalPreparedWorkspaceSource>,
    ) -> Result<RetainedStoragePublication, Error> {
        let inventory = self.inventory.as_mut().expect("one terminal inventory");
        inventory
            .publication_custody()
            .map_err(Error::PrefillControl)?;
        self.entries = if fixed.is_some() || prepared.is_some() {
            let entries = inventory.original_publication_entries(scope.pool(), fixed, prepared)?;
            self.original = Some(UnquotedOriginalSlotSources::default());
            entries
        } else {
            let (entries, original) = inventory.source_entries(scope.pool(), None)?;
            self.original = Some(original);
            entries
        };
        // An explicit immutable host owner is a separate positive source proof.
        // Recheck BOTH sides of that recorded alias. Unknown native backing on
        // its own never selects this route. Original input source validation is
        // unchanged: unknown or different original source profiles are rejected.
        for (identity, (bytes, array)) in inventory.array_entries() {
            if let Some((host_bytes, host)) = inventory.host_entry(identity) {
                require_same_capacity(*bytes, *host_bytes)?;
                let host_info = host.try_allocation_info().map_err(host_inspection_error)?;
                let array_info = array
                    .try_allocation_info()
                    .map_err(array_inspection_error)?;
                if host_info.identity() != *identity
                    || checked_bytes(host_info.bytes())? != *bytes
                    || !array_info.is_some_and(|info| {
                        info.identity() == *identity && info.bytes() == host_info.bytes()
                    })
                {
                    return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
                }
            }
        }
        self.entries.retain(|(key, _)| match key {
            StorageIdentity::Native(identity) => {
                inventory.array_entry(identity).is_none()
                    && inventory.host_entry(identity).is_none()
            }
            _ => true,
        });
        // One checked physical root per allocation. A retained Host owner is
        // sufficient even when no Array aliases it (the CPU General-copy case).
        // Shared Host/Array backing uses that same immutable owner once; the
        // paired identity/capacity check above still authenticates both sides.
        let roots = borrowed_native_roots(inventory);
        let attempt = self.attempt.as_mut().expect("claimed exact scope");
        attempt
            .publish(scope, roots, &self.entries)
            .map_err(|cause| {
                retained_failure_at(cause, controls.clone(), false, attempt.failure_site())
            })?;
        // The prepared registry remains the failure owner while each source
        // receives an independent clone. No map, key clone or result buffer is
        // allocated here, and a later attachment error keeps every original row.
        inventory.attach_prepared(
            |key| {
                self.entries
                    .iter()
                    .position(|(entry, _)| entry == key)
                    .and_then(|index| attempt.clone_source_for_attachment(index))
            },
            self.original.as_ref().expect("validated source inventory"),
            fixed,
            scope.pool().shared_storage_accounting_id(),
            false,
            false,
        )?;
        let registrations = attempt
            .take_remaining_sources()
            .expect("successful source transfer");
        let inventory = self.inventory.take().expect("successful inventory");
        Ok(inventory.finish_publication(
            || registrations,
            self.original.take().expect("source owner"),
            scope.pool().shared_storage_accounting_id(),
        ))
    }
}

pub(super) fn borrowed_native_roots(
    inventory: &RetainedStorage,
) -> impl Iterator<Item = NativeStorageRoot<'_>> {
    inventory
        .array_entries()
        .filter(|(identity, _)| inventory.host_entry(identity).is_none())
        .map(|(_, (_, array))| match array.canonical() {
            Some(cell) => NativeStorageRoot::CanonicalArray(cell),
            None => NativeStorageRoot::Array(array),
        })
        .chain(
            inventory
                .host_entries()
                .map(|(_, (_, host))| NativeStorageRoot::Host(host, host.attachment_receipt())),
        )
}

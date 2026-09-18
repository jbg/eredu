//! Publish charges onto physical allocations and closed shared host owners.

use super::*;
use crate::backend::managed_memory::NativeMemoryOwner;
use eredu_runtime::working_memory::WorkingMemoryFundingScope;

#[derive(Debug)]
struct PublishedStorage {
    domain: eredu_core::SharedStorageDomain,
    // Native arrays and host buffers are removed after successful attachment.
    // The remaining source/buffer roots retire outside native locks before
    // their registrations. Their weak identity tokens prevent address reuse.
    _non_native: RetainedStorage,
    _registrations: Vec<WorkingMemoryStorage<StorageIdentity>>,
    _original: UnquotedOriginalSlotSources,
    // Last: registrations and their Vec backing also belong to this Q hold.
    _metadata_custody: Option<eredu_runtime::working_memory::OriginalTextMetadataCustody>,
    _host: Option<eredu_core::HostPreparationAuthority>,
}

/// Source-owned roots remaining after native allocation charges are published.
/// This contains no native allocation retained by an attached charge and cannot
/// form an array -> charge -> array cycle. It does not clear an unquoted owner.
#[derive(Clone)]
pub(crate) struct RetainedStoragePublication(
    #[allow(dead_code)] std::rc::Rc<OrdinaryRetirement<PublishedStorage>>,
);

impl RetainedStoragePublication {
    /// Frames used by the read-only receipt lookup. The ordinary initial map
    /// already owns its nodes; lookup neither clones a key nor builds a table.
    pub(crate) fn attachment_lookup_control_bytes() -> Option<usize> {
        use std::mem::size_of;
        let parts = [
            size_of::<&Self>(),
            size_of::<&eredu_core::SharedStorageDomain>(),
            size_of::<safemlx::AllocationInfo>(),
            size_of::<safemlx::AllocationIdentity>(),
            size_of::<u64>(), // exact capacity forwarded to the scalar receipt lookup
            size_of::<Option<&NativeEntry<RetainedArray>>>(),
            size_of::<Option<&NativeEntry<RetainedHostBuffer>>>(),
            size_of::<bool>(), // Host receipt comparison
            size_of::<bool>(),
        ];
        parts.into_iter().try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }

    pub(crate) fn has_native_attachment(
        &self,
        domain: &eredu_core::SharedStorageDomain,
        allocation: safemlx::AllocationInfo,
    ) -> bool {
        self.has_native_attachment_facts(domain, allocation.identity(), allocation.bytes() as u64)
    }

    fn has_native_attachment_facts(
        &self,
        domain: &eredu_core::SharedStorageDomain,
        identity: safemlx::AllocationIdentity,
        capacity: u64,
    ) -> bool {
        self.0.domain.same_identity(domain)
            && (matches!(self.0._non_native.arrays.get(&identity),
                Some(NativeEntry::Attached(bytes)) if *bytes == capacity)
                || matches!(self.0._non_native.hosts.get(&identity),
                    Some(NativeEntry::Attached(bytes)) if *bytes == capacity))
    }
    /// All charges that could not attach to their physical owners must remain
    /// covered by this separate, still-retained publication. Source aliases in
    /// the inventory retain their own original construction accounts. An
    /// unquoted source population is outside this narrow comparison.
    pub(crate) fn can_retire_with(&self, retained: &Self) -> bool {
        !self.0._original.has_custody()
            && self.0._registrations.iter().all(|charge| {
                retained.0._registrations.iter()
                    .any(|other| charge.same_registered_storage(other))
            })
    }

    pub(crate) fn coverage_control_bytes() -> Option<usize> {
        use std::{mem::size_of, slice::Iter};
        let parts = [
            size_of::<(&Self, &Self)>(),
            size_of::<Option<&Self>>(),
            size_of::<bool>(),
            size_of::<[Iter<'static, WorkingMemoryStorage<StorageIdentity>>; 2]>(),
            size_of::<(&WorkingMemoryStorage<StorageIdentity>, &WorkingMemoryStorage<StorageIdentity>)>(),
            size_of::<Option<(&WorkingMemoryPool, &WorkingMemoryPool)>>(),
            size_of::<[Iter<'static, StorageIdentity>; 2]>(),
        ];
        parts.into_iter().try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
}

impl std::fmt::Debug for RetainedStoragePublication {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("RetainedStoragePublication")
            .field(&**self.0)
            .finish()
    }
}

impl RetainedStorage {
    /// Registers a complete existing inventory, attaching each native charge to
    /// precisely its physical backing. The domain owner must remain unquoted:
    /// publication alone proves neither total model coverage nor future work.
    ///
    /// Admission is atomic for the whole inventory. If a later native attachment
    /// fails, already attached charges remain valid and the caller's unquoted
    /// owner remains active; no partial publication authorizes quoted execution.
    pub(crate) fn publish_unquoted(
        self,
        owner: &NativeMemoryOwner,
    ) -> Result<RetainedStoragePublication, Error> {
        self.publication_custody().map_err(Error::PrefillControl)?;
        let (entries, original) = self.source_entries(owner.pool(), Some(owner))?;
        let registrations = owner.pool().register_storage_individually(entries);
        let registrations = match registrations {
            Ok(value) => value,
            Err(error) => {
                return Err(original_source_failure(
                    Error::Other(Box::new(error)),
                    original,
                ));
            }
        };
        self.attach_publication(
            registrations,
            original,
            None,
            owner.pool().shared_storage_domain(),
        )
    }

    /// Transfers a complete existing inventory from an active funded envelope
    /// to independent charges attached to its physical allocations. Previously
    /// registered aliases share their charge instead of consuming more funding.
    /// Inventory inspection never evaluates or polls native arrays.
    ///
    /// The caller must keep this scope through exact native settlement and all
    /// retained publication, then certify it only after those operations succeed.
    /// On any failure it must leave the scope uncertified: unattached charges
    /// return their credit to the still-active envelope when they retire, and an
    /// uncertified scope quarantines that envelope. Already attached charges
    /// remain on their backing. Partial attachment therefore grants no refund.
    ///
    /// Native sidecars contain only their per-allocation registration, never
    /// this inventory, its array/source roots, or the caller's funding scope.
    pub(crate) fn publish_funded(
        self,
        funding: &WorkingMemoryFundingScope,
    ) -> Result<RetainedStoragePublication, Error> {
        self.publication_custody().map_err(Error::PrefillControl)?;
        let (entries, original) = self.source_entries(funding.pool(), None)?;
        let registrations = funding.adopt_storage_individually(entries);
        let registrations = match registrations {
            Ok(value) => value,
            Err(error) => {
                return Err(original_source_failure(
                    Error::Other(Box::new(error)),
                    original,
                ));
            }
        };
        self.attach_publication(
            registrations,
            original,
            None,
            funding.pool().shared_storage_domain(),
        )
    }

    /// The model Work already validated both inventories before mutation.
    /// Recheck this private source at entry; its inline owner stays with Work.
    /// The existing inventory owns the matching token through all attachment.
    pub(crate) fn publish_funded_with_original_table(
        self,
        funding: &WorkingMemoryFundingScope,
        original: &eredu_runtime::working_memory::OriginalResidentResetSource,
    ) -> Result<RetainedStoragePublication, Error> {
        self.publish_funded_with_original_sources(funding, Some(original), None)
    }

    pub(crate) fn publish_funded_with_original_sources(
        self,
        funding: &WorkingMemoryFundingScope,
        original: Option<&eredu_runtime::working_memory::OriginalResidentResetSource>,
        prepared: Option<&eredu_runtime::input::OriginalPreparedWorkspaceSource>,
    ) -> Result<RetainedStoragePublication, Error> {
        self.publication_custody().map_err(Error::PrefillControl)?;
        let entries = self.original_publication_entries(funding.pool(), original, prepared)?;
        let registrations = funding
            .adopt_storage_individually(entries)
            .map_err(|e| Error::Other(Box::new(e)))?;
        self.attach_publication(
            registrations,
            UnquotedOriginalSlotSources::default(),
            original,
            funding.pool().shared_storage_domain(),
        )
    }

    // Validation precedes key cloning and every registration mutation. Original
    // table and B cache metadata already have their own accounts; keep their
    // actual inventory owners, but omit them from newly adopted entries.
    fn original_publication_entries(
        &self,
        pool: &WorkingMemoryPool,
        original: Option<&eredu_runtime::working_memory::OriginalResidentResetSource>,
        prepared: Option<&eredu_runtime::input::OriginalPreparedWorkspaceSource>,
    ) -> Result<Vec<(StorageIdentity, u64)>, Error> {
        self.validate_original_sources(pool, original, prepared)?;
        let mut entries = self.storage_entries()?;
        entries.retain(|(key, _)| {
            !original.is_some_and(|source| self.is_original_table_entry(source, key))
                && !prepared.is_some_and(|source| self.is_prepared_input_entry(source, key))
        });
        Ok(entries)
    }
    fn is_prepared_input_entry(
        &self,
        source: &eredu_runtime::input::OriginalPreparedWorkspaceSource,
        key: &StorageIdentity,
    ) -> bool {
        let StorageIdentity::HostMetadata(key) = key else {
            return false;
        };
        self.metadata_entries().any(|(identity, (_, metadata))| {
            identity.registry_key() == key && matches!(metadata,
                eredu_runtime::SharedHostMetadata::Input(input) if source.matches_cache_identity(input))
        })
    }

    // Called only after validate_original_table authenticated every exact table
    // and required the quoted outer root. This comparison allocates no key.
    fn is_original_table_entry(
        &self,
        source: &eredu_runtime::working_memory::OriginalResidentResetSource,
        key: &StorageIdentity,
    ) -> bool {
        let StorageIdentity::HostMetadata(key) = key else {
            return false;
        };
        self.slot_entries().any(|(identity, (_, token))| {
            identity.registry_key() == key && source.same_constructor(token)
        })
    }

    fn attach_publication(
        mut self,
        mut registrations: BTreeMap<StorageIdentity, WorkingMemoryStorage<StorageIdentity>>,
        original: UnquotedOriginalSlotSources,
        fixed: Option<&eredu_runtime::working_memory::OriginalResidentResetSource>,
        domain: &eredu_core::SharedStorageDomain,
    ) -> Result<RetainedStoragePublication, Error> {
        if let Err(cause) = self.attach_prepared(
            |key| registrations.remove(key),
            &original,
            fixed,
            domain,
            true,
            false,
        ) {
            return Err(original_source_failure(cause, original));
        }
        Ok(self.finish_publication(|| registrations.into_values().collect(), original, domain))
    }

    // Shared ordinary/source attachment worker. The selected native route has
    // already consumed checked array witnesses; host/source policy stays here.
    fn attach_prepared(
        &mut self,
        mut registration: impl FnMut(&StorageIdentity) -> Option<WorkingMemoryStorage<StorageIdentity>>,
        original: &UnquotedOriginalSlotSources,
        fixed: Option<&eredu_runtime::working_memory::OriginalResidentResetSource>,
        domain: &eredu_core::SharedStorageDomain,
        arrays: bool,
        prepared_copy: bool,
    ) -> Result<(), Error> {
        if arrays {
            for (identity, (_, array)) in self.array_entries() {
                let charge = registration(&StorageIdentity::Native(*identity))
                    .expect("certified array has a registration");
                array.retain_allocation_owner(charge).map_err(|failure| {
                    let (error, _unattached) = failure.into_parts();
                    Error::from(error)
                })?;
            }
        }
        for (identity, (_, host)) in self.host_entries() {
            if prepared_copy {
                // The closed Copy publisher already authenticated each exact
                // completed Host destination and attached its prepared node.
                continue;
            }
            // A certified host/array alias already has its single physical
            // charge attached through the shared host-storage owner.
            if let Some(charge) = registration(&StorageIdentity::Native(*identity)) {
                host.retain_allocation_owner(charge).map_err(|failure| {
                    let (error, _unattached) = failure.into_parts();
                    Error::from(error)
                })?;
            }
        }
        for (identity, (_, metadata)) in self.metadata_entries() {
            if let eredu_runtime::SharedHostMetadata::Input(input) = metadata {
                if let Some(matches) = input.original_domain_matches(domain) {
                    if !matches {
                        return Err(Error::Other(Box::new(WorkingMemoryError::IdentityMismatch)));
                    }
                    // source_entries authenticated this exact B owner before
                    // registry mutation. It remains in _non_native below.
                    continue;
                }
            }
            let charge = registration(&StorageIdentity::HostMetadata(
                identity.registry_key().clone(),
            ))
            .expect("certified metadata has a registration");
            metadata
                .try_attach(domain, || {
                    Ok::<Box<dyn Send + Sync>, WorkingMemoryError>(Box::new(charge))
                })
                .map_err(|error| Error::Other(Box::new(error)))?;
        }
        for (identity, (_, token)) in self.slot_entries() {
            if fixed.is_some_and(|source| source.same_constructor(token))
                || original
                    .sources()
                    .iter()
                    .any(|source| source.metadata().same_storage(token))
            {
                // Exact runtime-authenticated original source. Its own account
                // already pays the table; no attachment or duplicate registry.
                continue;
            }
            let charge = registration(&StorageIdentity::HostMetadata(
                identity.registry_key().clone(),
            ))
            .expect("certified host slot extent has a registration");
            let attach = || Ok::<Box<dyn Send + Sync>, WorkingMemoryError>(Box::new(charge));
            if prepared_copy {
                token.try_attach_prepared_copy(domain, attach)
            } else {
                token.try_attach(domain, attach)
            }
            .map_err(|error| Error::Other(Box::new(error)))?;
        }
        for (identity, (_, plan)) in self.capture_entries() {
            let charge = registration(&StorageIdentity::CapturePlan(identity.clone()))
                .expect("certified capture plan has a registration");
            plan.try_attach(domain, || {
                Ok::<Box<dyn Send + Sync>, WorkingMemoryError>(Box::new(charge))
            })
            .map_err(|error| Error::Other(Box::new(error)))?;
        }
        Ok(())
    }

    fn finish_publication(
        mut self,
        registrations: impl FnOnce() -> Vec<WorkingMemoryStorage<StorageIdentity>>,
        original: UnquotedOriginalSlotSources,
        domain: &eredu_core::SharedStorageDomain,
    ) -> RetainedStoragePublication {
        let metadata_custody = self
            .publication_custody()
            .expect("validated publication role");
        self.finish_publication_with_custody(registrations, original, metadata_custody, None, domain)
    }
    fn finish_publication_with_custody(
        mut self,
        registrations: impl FnOnce() -> Vec<WorkingMemoryStorage<StorageIdentity>>,
        original: UnquotedOriginalSlotSources,
        metadata_custody: Option<eredu_runtime::working_memory::OriginalTextMetadataCustody>,
        host: Option<eredu_core::HostPreparationAuthority>,
        domain: &eredu_core::SharedStorageDomain,
    ) -> RetainedStoragePublication {
        if let Some(original) = self.original.as_mut() {
            original.finish();
        }
        // Reuse the already constructed map nodes. Only a successful complete
        // attachment pass reaches here, and allocation generations never reuse.
        // A receipt retains neither an Array nor a registration/payload pin.
        for entry in self.arrays.values_mut() {
            *entry = NativeEntry::Attached(entry.bytes());
        }
        for entry in self.hosts.values_mut() {
            *entry = NativeEntry::Attached(entry.bytes());
        }
        // The actual immutable owners now retain only their per-source charge.
        // No publication record keeps a root that points back to that record.
        self.metadata.retain(|_, (_, metadata)| {
            matches!(metadata,
            eredu_runtime::SharedHostMetadata::Input(input) if input.original_source().is_some())
        });
        // Earlier shared-plan aliases now retain this charge directly. The
        // sidecar owns only a payload-free key, never the plan or this inventory.
        self.capture_plans.clear();
        // The table and any earlier escaped tokens now share its attachment.
        // No registration owns a token or can recover the table payload.
        self.slot_metadata.clear();
        RetainedStoragePublication(std::rc::Rc::new(OrdinaryRetirement::new(
            PublishedStorage {
                domain: domain.clone(),
                _non_native: self,
                // Detached host/source charges remain here. Native attachment
                // receipts above hold only scalar generation/capacity facts.
                _registrations: registrations(),
                _original: original,
                _metadata_custody: metadata_custody,
                _host: host,
            },
        )))
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod metadata_tests;

#[cfg(test)]
mod slot_metadata_tests;

#[cfg(test)]
mod capture_plan_tests;

mod native;
pub(crate) use native::PendingNativePublication;

fn publication_control_bytes(rows: usize) -> Option<u64> {
    use std::{alloc::Layout, mem::size_of};
    // Same pinned RcInner recipe qualified by the selected bank's owning query.
    // OrdinaryRetirement holds the raw inventory after this Rc is deallocated;
    // its unboxing retires the node before the extracted inventory releases Q.
    let shared = Layout::new::<[usize; 2]>()
        .align_to(2)
        .ok()?
        .pad_to_align()
        .extend(Layout::new::<OrdinaryRetirement<PublishedStorage>>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    let fixed = [
        shared,
        size_of::<PublishedStorage>(),
        size_of::<std::collections::btree_map::ValuesMut<'_, safemlx::AllocationIdentity, NativeEntry<RetainedHostBuffer>>>(),
        size_of::<&mut NativeEntry<RetainedHostBuffer>>(),
        size_of::<NativeEntry<RetainedHostBuffer>>(),
        size_of::<u64>(),
        size_of::<RetainedStoragePublication>(),
        size_of::<std::rc::Rc<OrdinaryRetirement<PublishedStorage>>>(),
        size_of::<Vec<WorkingMemoryStorage<StorageIdentity>>>(),
        size_of::<Box<dyn Send + Sync>>(),
        size_of::<StorageIdentity>(),
        size_of::<Result<bool, eredu_core::SharedStorageAttachmentError<WorkingMemoryError>>>(),
    ];
    let boxes = Layout::array::<WorkingMemoryStorage<StorageIdentity>>(rows)
        .ok()?
        .size();
    u64::try_from(
        fixed
            .into_iter()
            .try_fold(std::mem::size_of_val(&fixed), usize::checked_add)?
            .checked_add(boxes)?,
    )
    .ok()?
    .checked_add(OrdinaryRetirement::<PublishedStorage>::control_bytes()?)
}

mod copy;
pub(crate) use copy::retain_failure as retain_copy_publication_failure;
pub(crate) use copy::{CopyPublicationLayout, PendingCopyPublication};

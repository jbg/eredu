//! Same canonical registry, with authenticated cold source coverage.
use super::*;
use crate::working_memory::gguf_source::{GgufSourceStorageKey, SourceInventoryOrigin};

impl MemoryLedger {
    /// Read-only validation of an existing checkpoint payload. An original
    /// constructor keeps its same-pool custody checks. A load-time ordinary
    /// source must already have this exact fully charged canonical registration.
    /// No row, pin, credit, constructor authority or registration is created.
    pub fn validate_retained_source_inventory<K>(
        &self,
        key: &K,
        bytes: u64,
    ) -> Result<(), WorkingMemoryError>
    where
        K: Ord + Send + Sync + 'static + GgufSourceStorageKey,
    {
        let identity = key
            .gguf_source_identity()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if SourceInventoryOrigin::has_original_constructor(identity) {
            return self.validate_original_source_inventory(identity, bytes);
        }
        self.validate_registered_ordinary_source(key, bytes)
    }

    pub(in crate::working_memory) fn validate_registered_ordinary_source<K>(
        &self,
        key: &K,
        bytes: u64,
    ) -> Result<(), WorkingMemoryError>
    where
        K: Ord + Send + Sync + 'static,
    {
        let usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let registry = usage
            .storage
            .get(&TypeId::of::<K>())
            .and_then(|r| r.downcast_ref::<Registry<K>>())
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let (_, entry) = registry
            .locate(key)
            .ok_or(WorkingMemoryError::UnknownBound)?;
        same_capacity(entry.bytes, bytes)?;
        validate_entry_origin(entry, &usage)?;
        if entry.owners == 0 || entry.prepaid.is_some() || entry.funding.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }

    /// Reused synchronous registry/source validation frames; no row allocation.
    pub fn retained_source_inventory_control_bytes<K>() -> Option<usize> {
        let frames = [
            std::mem::size_of::<(&MemoryLedger, &K, u64)>(),
            std::mem::size_of::<&eredu_checkpoint::store::SourceStorageIdentity>(),
            std::mem::size_of::<Option<&eredu_checkpoint::store::SourceStorageIdentity>>(),
            std::mem::size_of::<bool>() * 2,
            std::mem::size_of::<std::sync::MutexGuard<'static, super::super::Usage>>(),
            std::mem::size_of::<&Registry<K>>(),
            std::mem::size_of::<(EntryLocator, &Entry)>(),
            std::mem::size_of::<Result<(), WorkingMemoryError>>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }

    /// Registers the full physical inventory and recognizes only a built-in
    /// source constructor's actual same-pool prepaid portion. Other identities
    /// and foreign pools retain ordinary full-capacity accounting. No provider
    /// supplies byte credit; the key projection lends only its opaque identity.
    ///
    /// This is ordinary cold registry preparation, not permission for registry
    /// control allocation. Prepared rows and the existing namespace candidate
    /// keep all provider key construction/destruction outside Usage. Aliases
    /// retain their first canonical origin; an ordinary row is never promoted.
    pub(super) fn register_storage_with_gguf_sources_prepared<K>(
        &self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError>
    where
        K: Clone + Ord + Send + Sync + 'static + GgufSourceStorageKey,
    {
        let unique = physical::prepare(self, storage)?;
        let bytes = unique.iter().try_fold(0u64, |sum, (_, allocation)| {
            sum.checked_add(allocation.capacity_bytes())
        });
        let mut sources = Vec::with_capacity(unique.len());
        for (key, allocation) in &unique {
            if let Some(identity) = key.gguf_source_identity() {
                if allocation.placement() != self.0.host_placement.as_ref() {
                    return Err(WorkingMemoryError::StoragePlacementMismatch);
                }
                if let Some(origin) =
                    SourceInventoryOrigin::inspect(identity, allocation.capacity_bytes(), self)?
                {
                    sources.push((key.clone(), origin));
                }
            }
        }
        let mut registration = WorkingMemoryStorage::pending_domains(
            unique.iter().map(|(key, _)| key.clone()).collect(),
            bytes,
        );
        Arc::get_mut(&mut registration.0).unwrap().preparation = Some(host.clone());
        self.commit_physical_inventory(unique, None, 0, sources, host)?;
        registration.activate(self.clone(), None);
        Ok(registration)
    }
}

#[cfg(test)]
impl MemoryLedger {
    pub(crate) fn register_storage_with_gguf_sources<K>(
        &self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError>
    where
        K: Clone + Ord + Send + Sync + 'static + GgufSourceStorageKey,
    {
        let inputs: Vec<_> = storage.into_iter().collect();
        StoragePublicationLayout::new(inputs.len())?
            .fund(self)?
            .register_storage_with_gguf_sources(inputs)
    }
}

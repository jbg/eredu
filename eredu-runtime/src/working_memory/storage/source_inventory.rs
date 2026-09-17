//! Same canonical registry, with authenticated cold source coverage.
use super::*;
use crate::working_memory::gguf_source::{GgufSourceStorageKey, SourceInventoryOrigin};

struct Row<K> {
    key: K,
    registry_key: Option<K>,
    bytes: u64,
    origin: Option<SourceInventoryOrigin>,
    locator: Option<EntryLocator>,
}
impl WorkingMemoryPool {
    /// Read-only validation of an existing checkpoint payload. An original
    /// constructor keeps its same-pool custody checks. A load-time ordinary
    /// source must already have this exact fully charged canonical registration.
    /// No row, pin, credit, constructor authority or registration is created.
    pub fn validate_retained_source_inventory<K>(
        &self, key: &K, bytes: u64,
    ) -> Result<(), WorkingMemoryError>
    where K: Ord + Send + Sync + 'static + GgufSourceStorageKey {
        let identity = key.gguf_source_identity().ok_or(WorkingMemoryError::IdentityMismatch)?;
        if SourceInventoryOrigin::has_original_constructor(identity) {
            return self.validate_original_source_inventory(identity, bytes);
        }
        self.validate_registered_ordinary_source(key, bytes)
    }

    pub(in crate::working_memory) fn validate_registered_ordinary_source<K>(
        &self, key: &K, bytes: u64,
    ) -> Result<(), WorkingMemoryError>
    where K: Ord + Send + Sync + 'static {
        let usage = self.0.usage.lock().map_err(|_| WorkingMemoryError::Poisoned)?;
        let registry = usage.storage.get(&TypeId::of::<K>())
            .and_then(|r| r.downcast_ref::<Registry<K>>())
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let (_, entry) = registry.locate(key).ok_or(WorkingMemoryError::UnknownBound)?;
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
            std::mem::size_of::<(&WorkingMemoryPool, &K, u64)>(),
            std::mem::size_of::<&eredu_checkpoint::store::SourceStorageIdentity>(),
            std::mem::size_of::<Option<&eredu_checkpoint::store::SourceStorageIdentity>>(),
            std::mem::size_of::<bool>() * 2,
            std::mem::size_of::<std::sync::MutexGuard<'static, super::super::Usage>>(),
            std::mem::size_of::<&Registry<K>>(),
            std::mem::size_of::<(EntryLocator, &Entry)>(),
            std::mem::size_of::<Result<(), WorkingMemoryError>>(),
        ];
        frames.into_iter().try_fold(std::mem::size_of_val(&frames), usize::checked_add)
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
    pub fn register_storage_with_gguf_sources<K>(
        &self,
        storage: impl IntoIterator<Item = (K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError>
    where
        K: Clone + Ord + Send + Sync + 'static + GgufSourceStorageKey,
    {
        let (unique, bytes) = prepare_inventory(storage)?;
        let keys = unique.keys().cloned().collect();
        let mut registration = WorkingMemoryStorage::pending(keys, bytes);
        if unique.is_empty() {
            registration.activate(self.clone(), None);
            return Ok(registration);
        }
        let mut rows = Vec::with_capacity(unique.len());
        for (key, bytes) in unique {
            let origin = match key.gguf_source_identity() {
                None => None,
                Some(identity) => SourceInventoryOrigin::inspect(identity, bytes, self)?,
            };
            rows.push(Row {
                registry_key: Some(key.clone()),
                key,
                bytes,
                origin,
                locator: None,
            });
        }
        let mut namespace = Some(directory::PreparedNamespace::prepare::<K>(None));
        let mut node = Some(RegistryBatch::prepare_source_registration(rows.len()));
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let missing = usage.storage.get(&TypeId::of::<K>()).is_none();
        let registry = usage
            .storage
            .get(&TypeId::of::<K>())
            .map(|r| {
                r.downcast_ref::<Registry<K>>()
                    .expect("typed source inventory")
            })
            .unwrap_or_else(|| namespace.as_ref().unwrap().registry::<K>());
        let mut incremental = 0u64;
        let mut new_rows = 0usize;
        for row in &mut rows {
            if let Some(origin) = &row.origin {
                origin.validate_pool(self)?;
            }
            if let Some((locator, entry)) = registry.locate(&row.key) {
                same_capacity(entry.bytes, row.bytes)?;
                validate_entry_origin(entry, &usage)?;
                if let (Some(origin), Some(prior)) = (&row.origin, &entry.prepaid) {
                    if !matches!(prior, prepaid::PrepaidStorageOrigin::Source(prior) if origin.same(prior))
                    {
                        return Err(WorkingMemoryError::IdentityMismatch);
                    }
                }
                entry
                    .owners
                    .checked_add(1)
                    .ok_or(WorkingMemoryError::Overflow)?;
                row.locator = Some(locator);
            } else {
                new_rows = new_rows
                    .checked_add(1)
                    .ok_or(WorkingMemoryError::Overflow)?;
                incremental = incremental
                    .checked_add(row.origin.as_ref().map_or(row.bytes, |o| o.residual()))
                    .ok_or(WorkingMemoryError::Overflow)?;
            }
        }
        let available = self.0.available(&usage, None)?;
        if incremental > available {
            return Err(WorkingMemoryError::BudgetExceeded {
                required_bytes: incremental,
                available_bytes: available,
            });
        }
        let registered = usage
            .registered
            .checked_add(incremental)
            .ok_or(WorkingMemoryError::Overflow)?;
        let used = self
            .0
            .existing
            .checked_add(registered)
            .and_then(|n| n.checked_add(usage.reserved))
            .ok_or(WorkingMemoryError::Overflow)?;
        // All fallible observations and provider comparisons precede ownership
        // movement. Commit writes scalar locators and prepared fixed slots only.
        if missing && new_rows != 0 {
            usage.storage.install(namespace.take().unwrap());
        }
        let registry = usage
            .storage
            .get_mut(&TypeId::of::<K>())
            .unwrap()
            .downcast_mut::<Registry<K>>()
            .unwrap();
        for (index, row) in rows.iter_mut().enumerate() {
            if let Some(locator) = row.locator {
                registry.at_mut(locator).owners += 1;
            } else {
                node.as_mut().unwrap().entries[index] = Some((
                    RegistryKey::Owned(row.registry_key.take().unwrap()),
                    Entry {
                        reset_layout_id: None,
                        bytes: row.bytes,
                        owners: 1,
                        funding: None,
                        prepaid: row.origin.take().map(prepaid::PrepaidStorageOrigin::Source),
                    },
                ));
            }
        }
        if new_rows != 0 {
            registry.link(node.take().unwrap());
        }
        usage.registered = registered;
        usage.peak = usage.peak.max(used);
        drop(usage);
        registration.activate(self.clone(), None);
        // Duplicates, unused candidates, node shells and their accounting-only
        // aliases retire here. Provider unwind sees an already active output.
        drop(rows);
        drop(node);
        drop(namespace);
        Ok(registration)
    }
}

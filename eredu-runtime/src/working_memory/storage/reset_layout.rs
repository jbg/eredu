//! One independently owned existing layout entry. No provider key escapes here.
use super::*;
use crate::{
    working_memory::{HostSlotStorageKey, Usage},
    HostMetadataKey,
};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Private constructors bind every function pointer to one concrete registry
/// namespace. None can call provider cloning/projection/comparison at retirement.
#[derive(Debug)]
pub(in crate::working_memory) struct ResetLayoutPin {
    pool: WorkingMemoryPool,
    identity: HostMetadataKey,
    bytes: u64,
    id: u64,
    namespace: TypeId,
    active: bool,
    acquire: fn(&mut Usage, u64, u64) -> Result<(), WorkingMemoryError>,
    retire: fn(&WorkingMemoryPool, u64),
}
impl ResetLayoutPin {
    /// Locate an existing canonical entry using an actual source-derived key.
    /// The caller keeps this locator under the same Usage loan until commit.
    pub(in crate::working_memory) fn locate_existing<K: HostSlotStorageKey>(
        usage: &Usage,
        key: &K,
        bytes: u64,
    ) -> Result<registry::EntryLocator, WorkingMemoryError> {
        let registry = usage
            .storage
            .get(&TypeId::of::<K>())
            .and_then(|value| value.downcast_ref::<Registry<K>>())
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let (locator, entry) = registry
            .locate(key)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if entry.bytes != bytes {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        validate_entry_origin(entry, usage)?;
        Ok(locator)
    }

    /// Acquire both existing source entries after every budget/overflow check.
    /// A possible shared locator is counted twice in one scalar update. No
    /// provider operation or fallible work follows the first mutation.
    pub(in crate::working_memory) fn acquire_pair<K: HostSlotStorageKey>(
        table: &mut Self,
        layout: &mut Self,
        table_locator: registry::EntryLocator,
        layout_locator: registry::EntryLocator,
        usage: &mut Usage,
    ) -> Result<(), WorkingMemoryError> {
        if table.active
            || layout.active
            || table.namespace != TypeId::of::<K>()
            || layout.namespace != TypeId::of::<K>()
            || !table.pool.same_domain(&layout.pool)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let registry = usage
            .storage
            .get_mut(&TypeId::of::<K>())
            .expect("same-loan validated namespace")
            .downcast_mut::<Registry<K>>()
            .expect("same typed namespace");
        let shared = table_locator == layout_locator;
        let (table_id, table_owners) = {
            let entry = registry.at_mut(table_locator);
            if entry.bytes != table.bytes {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            (
                entry.reset_layout_id.unwrap_or(table.id),
                entry
                    .owners
                    .checked_add(if shared { 2 } else { 1 })
                    .ok_or(WorkingMemoryError::Overflow)?,
            )
        };
        let (layout_id, layout_owners) = {
            let entry = registry.at_mut(layout_locator);
            if entry.bytes != layout.bytes {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            if shared {
                (table_id, table_owners)
            } else {
                (
                    entry.reset_layout_id.unwrap_or(layout.id),
                    entry
                        .owners
                        .checked_add(1)
                        .ok_or(WorkingMemoryError::Overflow)?,
                )
            }
        };
        let entry = registry.at_mut(table_locator);
        entry.reset_layout_id = Some(table_id);
        entry.owners = table_owners;
        if !shared {
            let entry = registry.at_mut(layout_locator);
            entry.reset_layout_id = Some(layout_id);
            entry.owners = layout_owners;
        }
        table.id = table_id;
        table.active = true;
        layout.id = layout_id;
        layout.active = true;
        Ok(())
    }

    pub(in crate::working_memory) fn prepare<K: HostSlotStorageKey>(
        pool: &WorkingMemoryPool,
        identity: &HostMetadataKey,
        bytes: u64,
    ) -> Result<Self, WorkingMemoryError> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let id = NEXT
            .fetch_update(AtomicOrdering::Relaxed, AtomicOrdering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| WorkingMemoryError::Overflow)?;
        Ok(Self {
            pool: pool.clone(),
            identity: identity.clone(),
            bytes,
            id,
            namespace: TypeId::of::<K>(),
            active: false,
            acquire: acquire::<K>,
            retire: retire::<K>,
        })
    }
    /// Pins one exact source table already present in this typed namespace.
    /// Preparation and eventual retirement occur outside the Usage loan.
    pub(in crate::working_memory) fn pin_existing_host<K: HostSlotStorageKey>(
        pool: &WorkingMemoryPool,
        metadata: &crate::HostSlotMetadata,
    ) -> Result<Self, WorkingMemoryError> {
        let bytes = metadata
            .capacity_bytes()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let identity = metadata.identity().registry_key();
        let key =
            K::from_host_slot_identity(identity.clone()).ok_or(WorkingMemoryError::UnknownBound)?;
        if key.host_slot_identity() != Some(identity) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut pin = Self::prepare::<K>(pool, identity, bytes)?;
        {
            let mut usage = pool
                .0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            let locator = Self::locate_existing::<K>(&usage, &key, bytes)?;
            let entry = usage
                .storage
                .get_mut(&TypeId::of::<K>())
                .unwrap()
                .downcast_mut::<Registry<K>>()
                .unwrap()
                .at_mut(locator);
            let owners = entry
                .owners
                .checked_add(1)
                .ok_or(WorkingMemoryError::Overflow)?;
            let id = entry.reset_layout_id.unwrap_or(pin.id);
            let entry = usage
                .storage
                .get_mut(&TypeId::of::<K>())
                .unwrap()
                .downcast_mut::<Registry<K>>()
                .unwrap()
                .at_mut(locator);
            entry.reset_layout_id = Some(id);
            entry.owners = owners;
            pin.id = id;
            pin.active = true;
        }
        Ok(pin)
    }

    pub(in crate::working_memory) fn prepare_again<K: HostSlotStorageKey>(
        &self,
        pool: &WorkingMemoryPool,
        identity: &HostMetadataKey,
        bytes: u64,
    ) -> Result<Self, WorkingMemoryError> {
        if !self.active
            || !self.pool.same_domain(pool)
            || self.namespace != TypeId::of::<K>()
            || &self.identity != identity
            || self.bytes != bytes
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(Self {
            pool: pool.clone(),
            identity: self.identity.clone(),
            bytes,
            id: self.id,
            namespace: self.namespace,
            active: false,
            acquire: self.acquire,
            retire: self.retire,
        })
    }
    pub(in crate::working_memory) fn acquire_registered<K: HostSlotStorageKey>(
        &mut self,
        registration: &WorkingMemoryStorage<K>,
        locator: registry::EntryLocator,
        usage: &mut Usage,
    ) -> Result<(), WorkingMemoryError> {
        if self.active
            || self.namespace != TypeId::of::<K>()
            || registration
                .0
                .pool
                .as_ref()
                .is_none_or(|pool| !pool.same_domain(&self.pool))
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let entry = usage
            .storage
            .get_mut(&TypeId::of::<K>())
            .expect("validated layout namespace")
            .downcast_mut::<Registry<K>>()
            .expect("same typed layout namespace")
            .at_mut(locator);
        assert_eq!(entry.bytes, self.bytes, "same-loan validated layout extent");
        let owners = entry
            .owners
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        let id = entry.reset_layout_id.unwrap_or(self.id);
        // All fallible and provider operations precede these scalar updates.
        entry.reset_layout_id = Some(id);
        entry.owners = owners;
        self.id = id;
        self.active = true;
        Ok(())
    }
    pub(in crate::working_memory) fn acquire_again(
        &mut self,
        usage: &mut Usage,
    ) -> Result<(), WorkingMemoryError> {
        if self.active {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        (self.acquire)(usage, self.id, self.bytes)?;
        self.active = true;
        Ok(())
    }
}
impl Drop for ResetLayoutPin {
    fn drop(&mut self) {
        if self.active {
            (self.retire)(&self.pool, self.id);
        }
    }
}
fn acquire<K: HostSlotStorageKey>(
    usage: &mut Usage,
    id: u64,
    bytes: u64,
) -> Result<(), WorkingMemoryError> {
    let registry = usage
        .storage
        .get(&TypeId::of::<K>())
        .and_then(|r| r.downcast_ref::<Registry<K>>())
        .ok_or(WorkingMemoryError::IdentityMismatch)?;
    let (locator, entry) = registry
        .locate_reset_layout(id)
        .ok_or(WorkingMemoryError::IdentityMismatch)?;
    if entry.bytes != bytes {
        return Err(WorkingMemoryError::IdentityMismatch);
    }
    validate_entry_origin(entry, usage)?;
    let owners = entry
        .owners
        .checked_add(1)
        .ok_or(WorkingMemoryError::Overflow)?;
    usage
        .storage
        .get_mut(&TypeId::of::<K>())
        .unwrap()
        .downcast_mut::<Registry<K>>()
        .unwrap()
        .at_mut(locator)
        .owners = owners;
    Ok(())
}
fn retire<K: HostSlotStorageKey>(pool: &WorkingMemoryPool, id: u64) {
    let retired = {
        let mut usage = funding::lock_for_retirement(pool);
        let registry = usage
            .storage
            .get_mut(&TypeId::of::<K>())
            .expect("live layout registry")
            .downcast_mut::<Registry<K>>()
            .expect("same typed layout registry");
        let mut retired = registry.retire_reset_layout(id);
        if registry.is_empty() {
            retired.namespace = usage.storage.remove(&TypeId::of::<K>());
        }
        retired
    };
    let released = retired
        .entry
        .as_ref()
        .and_then(|(_, entry)| entry.charged());
    // Canonical key, detached batch and original raw owner all retire outside
    // Usage, in their established order. A panic conservatively keeps credit.
    drop(retired);
    if let Some((bytes, origin)) = released {
        let mut usage = funding::lock_for_retirement(pool);
        if let Some(id) = origin {
            funding::retire_allocation(&mut usage, id, bytes);
        } else {
            usage.registered -= bytes;
        }
    }
}

pub(in crate::working_memory) fn retirement_control_bytes<K: HostSlotStorageKey>() -> Option<usize>
{
    registry::reset_layout_retirement_bytes::<K>()
}

pub(in crate::working_memory) fn child_pin_control_bytes<K: HostSlotStorageKey>() -> Option<usize> {
    [
        retirement_control_bytes::<K>()?,
        size_of::<K>(),
        size_of::<Option<K>>(),
        size_of::<ResetLayoutPin>(),
        size_of::<Result<ResetLayoutPin, WorkingMemoryError>>(),
        size_of::<registry::EntryLocator>(),
        size_of::<Result<registry::EntryLocator, WorkingMemoryError>>(),
        size_of::<std::sync::MutexGuard<'_, Usage>>(),
        size_of::<
            Result<
                std::sync::MutexGuard<'_, Usage>,
                std::sync::PoisonError<std::sync::MutexGuard<'_, Usage>>,
            >,
        >(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

pub(in crate::working_memory) fn pair_control_bytes() -> Option<usize> {
    [
        size_of::<(registry::EntryLocator, registry::EntryLocator)>(),
        size_of::<(
            Option<registry::EntryLocator>,
            Option<registry::EntryLocator>,
        )>(),
        size_of::<Result<registry::EntryLocator, WorkingMemoryError>>(),
        size_of::<(u64, usize)>(),
        size_of::<(u64, usize)>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

#[cfg(test)]
mod pair_tests {
    use super::*;
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct Key(HostMetadataKey);
    impl HostSlotStorageKey for Key {
        fn host_slot_identity(&self) -> Option<&HostMetadataKey> {
            Some(&self.0)
        }
    }
    #[test]
    fn second_existing_pin_overflow_and_shared_entry_overflow_are_atomic() {
        for shared in [false, true] {
            let pool = WorkingMemoryPool::new(10_000, 0).unwrap();
            let left = crate::HostSlotTable::new(vec![1_u32].into_boxed_slice());
            let right = crate::HostSlotTable::new(vec![2_u32].into_boxed_slice());
            let left_key = Key(left.metadata().identity().registry_key().clone());
            let right_key = if shared {
                left_key.clone()
            } else {
                Key(right.metadata().identity().registry_key().clone())
            };
            let bytes = left.metadata().capacity_bytes().unwrap();
            let registration = pool
                .register_storage([(left_key.clone(), bytes), (right_key.clone(), bytes)])
                .unwrap();
            let mut table = ResetLayoutPin::prepare::<Key>(&pool, &left_key.0, bytes).unwrap();
            let mut layout = ResetLayoutPin::prepare::<Key>(&pool, &right_key.0, bytes).unwrap();
            {
                let mut usage = pool.0.usage.lock().unwrap();
                let a = ResetLayoutPin::locate_existing::<Key>(&usage, &left_key, bytes).unwrap();
                let b = ResetLayoutPin::locate_existing::<Key>(&usage, &right_key, bytes).unwrap();
                let registry = usage
                    .storage
                    .get_mut(&TypeId::of::<Key>())
                    .unwrap()
                    .downcast_mut::<Registry<Key>>()
                    .unwrap();
                let original_a = registry.at_mut(a).owners;
                let original_b = registry.at_mut(b).owners;
                registry.at_mut(b).owners = if shared { usize::MAX - 1 } else { usize::MAX };
                assert!(matches!(
                    ResetLayoutPin::acquire_pair::<Key>(&mut table, &mut layout, a, b, &mut usage),
                    Err(WorkingMemoryError::Overflow)
                ));
                let registry = usage
                    .storage
                    .get_mut(&TypeId::of::<Key>())
                    .unwrap()
                    .downcast_mut::<Registry<Key>>()
                    .unwrap();
                assert!(registry.at_mut(a).reset_layout_id.is_none());
                assert!(registry.at_mut(b).reset_layout_id.is_none());
                if !shared {
                    assert_eq!(registry.at_mut(a).owners, original_a);
                }
                registry.at_mut(b).owners = original_b;
                assert!(!table.active && !layout.active);
            }
            drop((table, layout, registration));
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}

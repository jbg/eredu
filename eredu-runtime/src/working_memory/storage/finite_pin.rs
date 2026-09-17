//! Finite existing-only pin constructor shared with ordinary pin validation.
use super::*;
use crate::working_memory::qualified_storage;
use std::{alloc::Layout, mem::size_of};

pub(super) struct PinOrdinal {
    bytes: u64,
    ordinal: EntryLocator,
}
pub(super) fn ordinals(
    capacities: impl IntoIterator<Item = u64>,
    slots: usize,
    exact: bool,
) -> Result<Vec<PinOrdinal>, WorkingMemoryError> {
    let mut rows = qualified_storage::vector(slots, exact)?;
    for bytes in capacities {
        if rows.len() == slots {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        rows.push(PinOrdinal {
            bytes,
            ordinal: EntryLocator::Legacy(0),
        });
    }
    Ok(rows)
}

// Incoming key/capacity vector is priced by the binding producer. The key moves
// into this finite destination without Clone; no key payload estimate is inferred.
pub(in crate::working_memory) fn construction_bytes<K: Ord + Send + 'static>(
    slots: usize,
) -> Result<usize, WorkingMemoryError> {
    let keys = Layout::array::<K>(slots)
        .map_err(|_| WorkingMemoryError::Overflow)?
        .size();
    let rows = Layout::array::<PinOrdinal>(slots)
        .map_err(|_| WorkingMemoryError::Overflow)?
        .size();
    let shared = usize::try_from(qualified_storage::shared_bytes::<Registration<K>>()?)
        .map_err(|_| WorkingMemoryError::Overflow)?;
    let key_control = usize::try_from(qualified_storage::vector_control_bytes::<K>()?)
        .map_err(|_| WorkingMemoryError::Overflow)?;
    let row_control = usize::try_from(qualified_storage::vector_control_bytes::<PinOrdinal>()?)
        .map_err(|_| WorkingMemoryError::Overflow)?;
    [
        keys,
        rows,
        shared,
        key_control,
        row_control,
        size_of::<Registration<K>>(),
        size_of::<WorkingMemoryStorage<K>>(),
        size_of::<StorageOwner<K>>(),
        size_of::<Arc<Registration<K>>>(),
        size_of::<Option<Registration<K>>>(),
        size_of::<Vec<(K, u64)>>(),
        size_of::<std::vec::IntoIter<(K, u64)>>(),
        size_of::<Option<(K, u64)>>(),
        size_of::<(
            &WorkingMemoryPool,
            &mut WorkingMemoryStorage<K>,
            &mut [PinOrdinal],
        )>(),
        size_of::<std::sync::MutexGuard<'static, super::super::Usage>>(),
        size_of::<Option<registry::RetiredEntry<K>>>(),
        size_of::<Option<(u64, Option<u64>)>>(),
        size_of::<Result<WorkingMemoryStorage<K>, WorkingMemoryError>>(),
        size_of::<Result<(), WorkingMemoryError>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .ok_or(WorkingMemoryError::Overflow)
}

impl WorkingMemoryPool {
    pub(in crate::working_memory) fn pin_registered_storage_owned<K: Ord + Send + 'static>(
        &self,
        mut inputs: Vec<(K, u64)>,
        exact: bool,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        // All comparisons and duplicate key retirement precede the Usage loan.
        inputs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        for pair in inputs.windows(2) {
            if pair[0].0.cmp(&pair[1].0).is_eq() {
                same_capacity(pair[0].1, pair[1].1)
                    .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
            }
        }
        inputs.dedup_by(|a, b| a.0.cmp(&b.0).is_eq());
        let bytes = inputs.iter().try_fold(0u64, |sum, (_, bytes)| {
            sum.checked_add(*bytes).ok_or(WorkingMemoryError::Overflow)
        })?;
        let mut keys = qualified_storage::vector(inputs.len(), exact)?;
        let mut rows = ordinals(inputs.iter().map(|(_, bytes)| *bytes), inputs.len(), exact)?;
        for (key, _) in inputs {
            keys.push(key);
        }
        let mut registration = WorkingMemoryStorage::pending(keys, bytes);
        self.commit_existing_pin(&mut registration, &mut rows)?;
        Ok(registration)
    }

    // Both ordinary/cloned and prepared/moved inventories use this atomic
    // existing-only worker. No owned storage construction, key Clone or Drop
    // occurs after the Usage loan. Provider comparisons are separate source
    // obligations and finish before the first increment.
    pub(super) fn commit_existing_pin<K: Ord + Send + 'static>(
        &self,
        storage: &mut WorkingMemoryStorage<K>,
        rows: &mut [PinOrdinal],
    ) -> Result<(), WorkingMemoryError> {
        let registration = Arc::get_mut(&mut storage.0)
            .filter(|r| r.pool.is_none() && r.keys.len() == rows.len())
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let retained_pool = self.clone();
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if !registration.keys.is_empty() {
            let registry = usage
                .storage
                .get(&TypeId::of::<K>())
                .and_then(|r| r.downcast_ref::<Registry<K>>())
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            for (key, row) in registration.keys.iter().zip(rows.iter_mut()) {
                let (ordinal, entry) = registry
                    .locate(key)
                    .ok_or(WorkingMemoryError::IdentityMismatch)?;
                same_capacity(entry.bytes, row.bytes)
                    .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
                entry
                    .owners
                    .checked_add(1)
                    .ok_or(WorkingMemoryError::Overflow)?;
                row.ordinal = ordinal;
            }
            for (i, row) in rows.iter().enumerate() {
                if rows[..i].iter().any(|prior| prior.ordinal == row.ordinal) {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
            }
            // Keep the existing pin's domain ceiling/overflow validation, but
            // no new physical charge, registry entry or namespace is constructed.
            self.0.available(&usage, None)?;
            let registry = usage
                .storage
                .get_mut(&TypeId::of::<K>())
                .and_then(|r| r.downcast_mut::<Registry<K>>())
                .expect("validated pin namespace");
            for row in rows {
                registry.at_mut(row.ordinal).owners += 1;
            }
        }
        registration.pool = Some(retained_pool);
        drop(usage);
        Ok(())
    }
}

/// Owning constructor layout for a counted existing-only pin. Input key/value
/// storage and key payloads belong to the caller; keys move without Clone.
#[derive(Debug)]
pub struct ExistingStoragePinLayout<K: Ord + Send + 'static> {
    maximum: usize,
    bytes: usize,
    marker: std::marker::PhantomData<fn() -> K>,
}
impl<K: Ord + Send + 'static> ExistingStoragePinLayout<K> {
    /// Qualifies the actual fresh vector/shared-shell producer before allocation.
    pub fn new(maximum: usize) -> Result<Self, WorkingMemoryError> {
        let bytes = construction_bytes::<K>(maximum)?
            .checked_add(size_of::<Self>())
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            maximum,
            bytes,
            marker: std::marker::PhantomData,
        })
    }
    /// Final key/ordinal arrays, registration shell and named control frames.
    pub fn requested_bytes(&self) -> usize {
        self.bytes
    }
    /// Consumes caller-owned finite inputs under its already accepted host plan.
    /// No registration is created for missing keys and no physical bytes charged.
    pub fn construct(
        self,
        pool: &WorkingMemoryPool,
        inputs: Vec<(K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        if inputs.len() > self.maximum {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        pool.pin_registered_storage_owned(inputs, true)
    }
}

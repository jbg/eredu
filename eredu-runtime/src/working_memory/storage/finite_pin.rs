//! Finite existing-only pin constructor shared with ordinary pin validation.
use super::*;
use crate::working_memory::qualified_storage;
use std::{alloc::Layout, mem::size_of};

pub(super) struct PinOrdinal {
    bytes: u64,
    placement: Option<Arc<eredu_core::MemoryPlacement>>,
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
            placement: None,
            ordinal: EntryLocator::Fixed { batch: 0, slot: 0 },
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
            &MemoryLedger,
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

impl MemoryLedger {
    /// Host controls for a capture source and its native backing-control row.
    /// The original capture population includes this allowance per transfer.
    /// This quotation neither pins storage nor grants execution permission.
    pub fn capture_source_pin_control_bytes<K: Ord + Send + 'static>()
    -> Result<u64, WorkingMemoryError> {
        let frames = [
            construction_bytes::<K>(2)?,
            Layout::array::<(K, u64)>(2)
                .map_err(|_| WorkingMemoryError::Overflow)?
                .size(),
            size_of::<[Option<(K, u64)>; 2]>(),
            size_of::<crate::working_memory::OriginalTextMetadataCustody>(),
            eredu_core::HostPreparationAuthority::retention_bytes::<
                crate::working_memory::OriginalTextMetadataCustody,
            >()
            .ok_or(WorkingMemoryError::Overflow)?,
            size_of::<eredu_core::HostPreparationAuthority>(),
        ];
        let bytes = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        u64::try_from(bytes).map_err(|_| WorkingMemoryError::Overflow)
    }

    // Only a consumed, authenticated capture claim may construct this paid pin.
    // Its actual storage keys remain subject to the shared registry transaction.
    pub(in crate::working_memory) fn pin_original_capture_source<K: Ord + Send + 'static>(
        &self,
        native: &WorkingMemoryFundingScope,
        custody: &crate::working_memory::OriginalTextMetadataCustody,
        source: [Option<(K, u64)>; 2],
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        custody.validate_capture_source_pin(native)?;
        if !self.same_ledger(native.pool()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let host = eredu_core::HostPreparationAuthority::retain(custody.clone());
        let mut inputs = qualified_storage::vector(2, true)?;
        inputs.extend(source.into_iter().flatten());
        let mut storage = self.pin_registered_storage_owned(inputs, true)?;
        Arc::get_mut(&mut storage.0)
            .expect("private original capture pin")
            .preparation = Some(host);
        Ok(storage)
    }

    /// Pins complete authenticated backing descriptors through the same atomic
    /// registry transaction. Conflicting placement or capacity rejects all pins.
    #[cfg(test)]
    pub(crate) fn pin_registered_storage_with_placement<K: Ord + Send + 'static>(
        &self,
        storage: impl IntoIterator<Item = (K, StorageAllocation)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        let entries: Vec<_> = storage.into_iter().collect();
        StoragePublicationLayout::new(entries.len())?
            .fund(self)?
            .pin_registered_storage_with_placement(entries)
    }
    pub(in crate::working_memory) fn pin_registered_storage_placed_owned<
        K: Ord + Send + 'static,
    >(
        &self,
        mut inputs: Vec<(K, StorageAllocation)>,
        exact: bool,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        inputs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        for (_, allocation) in &inputs {
            allocation.placement().validate(self.topology())?;
        }
        for pair in inputs.windows(2) {
            if pair[0].0 == pair[1].0 && pair[0].1 != pair[1].1 {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
        }
        inputs.dedup_by(|a, b| a.0 == b.0);
        let bytes = inputs.iter().try_fold(0u64, |sum, (_, allocation)| {
            sum.checked_add(allocation.capacity_bytes())
        });
        let mut keys = qualified_storage::vector(inputs.len(), exact)?;
        let mut rows = qualified_storage::vector(inputs.len(), exact)?;
        for (key, allocation) in inputs {
            keys.push(key);
            rows.push(PinOrdinal {
                bytes: allocation.capacity_bytes(),
                placement: Some(allocation.placement_handle()),
                ordinal: EntryLocator::Fixed { batch: 0, slot: 0 },
            });
        }
        let mut registration = WorkingMemoryStorage::pending_domains(keys, bytes);
        self.commit_existing_pin(&mut registration, &mut rows)?;
        drop(rows);
        Ok(registration)
    }

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
        let bytes = inputs
            .iter()
            .try_fold(0u64, |sum, (_, bytes)| sum.checked_add(*bytes));
        let mut keys = qualified_storage::vector(inputs.len(), exact)?;
        let mut rows = ordinals(inputs.iter().map(|(_, bytes)| *bytes), inputs.len(), exact)?;
        for (key, _) in inputs {
            keys.push(key);
        }
        let mut registration = WorkingMemoryStorage::pending_domains(keys, bytes);
        self.commit_existing_pin(&mut registration, &mut rows)?;
        drop(rows);
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
                validate_entry_origin(entry, &usage)?;
                same_capacity(entry.bytes, row.bytes)
                    .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
                if row
                    .placement
                    .as_ref()
                    .is_some_and(|placement| placement != &entry.placement)
                {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
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
            for (slot, (domain, _)) in self.topology().domains().enumerate() {
                let current = self.0.domains[slot]
                    .existing
                    .checked_add(usage.domains[slot].reserved)
                    .and_then(|bytes| bytes.checked_add(usage.domains[slot].registered))
                    .ok_or(WorkingMemoryError::Overflow)?;
                self.0
                    .domain_capacity(&usage, domain, None)?
                    .check(domain, current, 0)?;
            }
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
        pool: &MemoryLedger,
        inputs: Vec<(K, u64)>,
    ) -> Result<WorkingMemoryStorage<K>, WorkingMemoryError> {
        if inputs.len() > self.maximum {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        pool.pin_registered_storage_owned(inputs, true)
    }
}

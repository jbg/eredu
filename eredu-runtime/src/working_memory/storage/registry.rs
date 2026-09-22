//! One namespace and ownership ledger for ordinary and preallocated entries.
use super::*;
use crate::working_memory::{OriginalHostMetadataCustody, funding::RawSpanHostOwner};

pub(super) struct Registry<K> {
    batches: Option<Box<RegistryBatch<K>>>,
}
// The slot allocation and link exist before Usage is borrowed. Each occupied
// slot is the canonical entry: there is no separate fixed-publication ledger.
pub(super) struct RegistryBatch<K> {
    pub(super) entries: RegistrySlots<K>,
    next: Option<Box<RegistryBatch<K>>>,
    // No C/source witness back-edge. Last-entry retirement extracts this hold
    // so keys and the complete node allocation retire before original custody.
    raw: Option<OriginalHostMetadataCustody>,
    native: Option<funding::native_partition::NativePartition>,
    preparation: Option<eredu_core::HostPreparationAuthority>,
}
// Ordinary publication preserves its existing boxed slice. The qualified
// native producer retains the exact Vec allocation, with no shrink/reallocation.
pub(super) enum RegistrySlots<K> {
    Ordinary(Box<[Option<(RegistryKey<K>, Entry)>]>),
    Native(Vec<Option<(RegistryKey<K>, Entry)>>),
}
impl<K> std::ops::Deref for RegistrySlots<K> {
    type Target = [Option<(RegistryKey<K>, Entry)>];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Ordinary(rows) => rows,
            Self::Native(rows) => rows,
        }
    }
}
impl<K> std::ops::DerefMut for RegistrySlots<K> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Ordinary(rows) => rows,
            Self::Native(rows) => rows,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::working_memory) enum EntryLocator {
    Fixed { batch: usize, slot: usize },
}
pub(super) struct RetiredEntry<K> {
    pub(super) entry: Option<(RegistryKey<K>, Entry)>,
    // Field order is part of the original custody boundary.
    _batch: Option<Box<RegistryBatch<K>>>,
    registration_key: Option<K>,
    // Last canonical-row removal detaches this shell before Usage unlock.
    pub(super) namespace: Option<super::directory::PreparedNamespace>,
    _raw: Option<OriginalHostMetadataCustody>,
    _native_batch: Option<funding::native_partition::NativePartition>,
    entry_origin: Option<super::prepaid::PrepaidStorageOrigin>,
    _preparation: Option<eredu_core::HostPreparationAuthority>,
}
impl<K> Drop for RetiredEntry<K> {
    fn drop(&mut self) {
        // Detach only custody. All keys and the actual Box retire before the
        // final partition, including reset-layout retirement and unwind.
        self.entry_origin = self
            .entry
            .as_mut()
            .and_then(|(_, entry)| entry.prepaid.take());
    }
}
impl<K> RetiredEntry<K> {
    pub(super) fn retire_with_key(mut self, key: K) {
        // Both canonical and retiring-registration keys precede raw custody,
        // even if either provider destructor unwinds. No allocation or lock.
        self.registration_key = Some(key);
    }
}
impl<K> RegistryBatch<K> {
    // Ordinary cold registration prepares its exact canonical row destinations
    // before Usage, with no original/raw metadata grant or fabricated partition.
    pub(super) fn cold_control_bytes(capacity: usize) -> Result<u64, WorkingMemoryError> {
        cold_metadata::vector_bytes::<Option<(RegistryKey<K>, Entry)>>(capacity)?
            .checked_add(
                u64::try_from(std::mem::size_of::<Self>())
                    .map_err(|_| WorkingMemoryError::Overflow)?,
            )
            .ok_or(WorkingMemoryError::Overflow)
    }
    pub(super) fn prepare_source_registration(slots: usize) -> Box<Self> {
        let mut entries = Vec::with_capacity(slots);
        entries.resize_with(slots, || None);
        Box::new(Self {
            entries: RegistrySlots::Ordinary(entries.into_boxed_slice()),
            next: None,
            raw: None,
            native: None,
            preparation: None,
        })
    }

    pub(super) fn prepare(slots: usize, raw: RawSpanHostOwner) -> Box<Self> {
        let mut entries = Vec::with_capacity(slots);
        entries.resize_with(slots, || None);
        Box::new(Self {
            entries: RegistrySlots::Ordinary(entries.into_boxed_slice()),
            next: None,
            raw: Some(raw.into()),
            native: None,
            preparation: None,
        })
    }
}
impl<K> RegistryBatch<K> {
    pub(super) fn prepare_native(
        slots: usize,
        partition: funding::native_partition::NativePartition,
    ) -> Box<Self> {
        let mut entries = Vec::with_capacity(slots);
        entries.resize_with(slots, || None);
        Box::new(Self {
            entries: RegistrySlots::Ordinary(entries.into_boxed_slice()),
            next: None,
            raw: None,
            native: Some(partition),
            preparation: None,
        })
    }
}
impl<K> RegistryBatch<K> {
    pub(super) fn prepare_native_exact(
        slots: usize,
        partition: funding::native_partition::NativePartition,
    ) -> Result<Box<Self>, WorkingMemoryError> {
        let mut entries = crate::working_memory::qualified_storage::vector(slots, true)?;
        entries.resize_with(slots, || None);
        Ok(Box::new(Self {
            entries: RegistrySlots::Native(entries),
            next: None,
            raw: None,
            native: Some(partition),
            preparation: None,
        }))
    }
}
impl<K> RegistryBatch<K> {
    pub(super) fn prepare_source_exact(
        slots: usize,
        raw: OriginalHostMetadataCustody,
    ) -> Result<Box<Self>, WorkingMemoryError> {
        let mut entries = crate::working_memory::qualified_storage::vector(slots, true)?;
        entries.resize_with(slots, || None);
        Ok(Box::new(Self {
            entries: RegistrySlots::Native(entries),
            next: None,
            raw: Some(raw),
            native: None,
            preparation: None,
        }))
    }
}
impl<K> RegistryBatch<K> {
    pub(super) fn prepare_copy_exact(
        slots: usize,
        host: &eredu_core::HostPreparationAuthority,
    ) -> Result<Box<Self>, WorkingMemoryError> {
        let mut entries = crate::working_memory::qualified_storage::vector(slots, true)?;
        entries.resize_with(slots, || None);
        Ok(Box::new(Self {
            entries: RegistrySlots::Native(entries),
            next: None,
            raw: None,
            native: None,
            preparation: Some(host.clone()),
        }))
    }
}
impl<K: Ord> Registry<K> {
    pub(super) fn new() -> Self {
        Self { batches: None }
    }
    pub(super) fn entries(&self) -> impl Iterator<Item = &Entry> + Clone {
        std::iter::successors(self.batches.as_deref(), |batch| batch.next.as_deref()).flat_map(
            |batch| {
                batch
                    .entries
                    .iter()
                    .filter_map(|slot| slot.as_ref().map(|(_, entry)| entry))
            },
        )
    }
    pub(super) fn locate(&self, key: &K) -> Option<(EntryLocator, &Entry)> {
        let mut batch = self.batches.as_deref();
        let mut index = 0usize;
        while let Some(current) = batch {
            for (slot, entry) in current.entries.iter().enumerate() {
                if let Some((candidate, entry)) = entry {
                    let borrowed: &K = candidate.borrow();
                    if borrowed.cmp(key) == Ordering::Equal {
                        return Some((EntryLocator::Fixed { batch: index, slot }, entry));
                    }
                }
            }
            index = index.checked_add(1).expect("registered batch population");
            batch = current.next.as_deref();
        }
        None
    }
    pub(super) fn at_mut(&mut self, locator: EntryLocator) -> &mut Entry {
        let EntryLocator::Fixed { batch, slot } = locator;
        let mut current = self.batches.as_deref_mut().expect("validated batch");
        for _ in 0..batch {
            current = current
                .next
                .as_deref_mut()
                .expect("validated batch ordinal");
        }
        &mut current.entries[slot]
            .as_mut()
            .expect("validated occupied slot")
            .1
    }
    pub(super) fn locate_reset_layout(&self, id: u64) -> Option<(EntryLocator, &Entry)> {
        let mut batch = self.batches.as_deref();
        let mut index = 0usize;
        while let Some(current) = batch {
            for (slot, entry) in current.entries.iter().enumerate() {
                if let Some((_, entry)) = entry {
                    if entry.reset_layout_id == Some(id) {
                        return Some((EntryLocator::Fixed { batch: index, slot }, entry));
                    }
                }
            }
            index = index.checked_add(1).expect("registered batch population");
            batch = current.next.as_deref();
        }
        None
    }
    // The runtime-issued reset identity requires no provider callback.
    pub(super) fn retire_reset_layout(&mut self, id: u64) -> RetiredEntry<K> {
        let (locator, _) = self.locate_reset_layout(id).expect("live reset layout pin");
        self.retire_at(locator)
    }
    pub(super) fn get(&self, key: &K) -> Option<&Entry> {
        self.locate(key).map(|(_, entry)| entry)
    }
    pub(super) fn get_mut(&mut self, key: &K) -> Option<&mut Entry> {
        let (locator, _) = self.locate(key)?;
        Some(self.at_mut(locator))
    }
    pub(super) fn is_empty(&self) -> bool {
        self.batches.is_none()
    }
    pub(super) fn link(&mut self, mut batch: Box<RegistryBatch<K>>) {
        debug_assert!(batch.next.is_none());
        debug_assert!(batch.entries.iter().any(Option::is_some));
        batch.next = self.batches.take();
        self.batches = Some(batch);
    }
    pub(super) fn retire_owner(&mut self, key: &K) -> RetiredEntry<K> {
        let (locator, _) = self.locate(key).expect("registered storage owner");
        self.retire_at(locator)
    }
    fn retire_at(&mut self, locator: EntryLocator) -> RetiredEntry<K> {
        let entry = self.at_mut(locator);
        assert_ne!(entry.owners, 0, "live canonical owner count");
        if entry.owners > 1 {
            entry.owners -= 1;
            return RetiredEntry {
                entry: None,
                _batch: None,
                registration_key: None,
                namespace: None,
                _raw: None,
                _native_batch: None,
                entry_origin: None,
                _preparation: None,
            };
        }
        let EntryLocator::Fixed { batch, slot } = locator;
        self.retire_fixed_last(batch, slot)
    }
    fn retire_fixed_last(&mut self, batch: usize, slot: usize) -> RetiredEntry<K> {
        let mut link = &mut self.batches;
        for _ in 0..batch {
            link = &mut link.as_mut().expect("located batch").next;
        }
        let current = link.as_mut().expect("located batch");
        let entry = current.entries[slot].take().expect("located occupied slot");
        if current.entries.iter().all(Option::is_none) {
            let mut retired = link.take().expect("located batch");
            *link = retired.next.take();
            let raw = retired.raw.take();
            let native = retired.native.take();
            let preparation = retired.preparation.take();
            RetiredEntry {
                entry: Some(entry),
                _batch: Some(retired),
                registration_key: None,
                namespace: None,
                _raw: raw,
                _native_batch: native,
                entry_origin: None,
                _preparation: preparation,
            }
        } else {
            RetiredEntry {
                entry: Some(entry),
                _batch: None,
                registration_key: None,
                namespace: None,
                _raw: None,
                _native_batch: None,
                entry_origin: None,
                _preparation: None,
            }
        }
    }
}

pub(super) fn reset_layout_retirement_bytes<K: Ord>() -> Option<usize> {
    [
        // Fixed-node helper result, registry result, and caller-owned retiree.
        std::mem::size_of::<RetiredEntry<K>>(),
        std::mem::size_of::<RetiredEntry<K>>(),
        std::mem::size_of::<RetiredEntry<K>>(),
        std::mem::size_of::<Option<(RegistryKey<K>, Entry)>>(),
        std::mem::size_of::<(RegistryKey<K>, Entry)>(),
        std::mem::size_of::<Option<(EntryLocator, &Entry)>>(),
        std::mem::size_of::<Option<(u64, Option<u64>)>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

#[cfg(test)]
mod tests;

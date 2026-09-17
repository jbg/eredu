//! One namespace and ownership ledger for ordinary and preallocated entries.
use super::*;
use crate::working_memory::{funding::RawSpanHostOwner, OriginalHostMetadataCustody};

pub(super) struct Registry<K> {
    legacy: BTreeMap<RegistryKey<K>, Entry>,
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
    Legacy(usize),
    Fixed { batch: usize, slot: usize },
}
pub(super) enum RegistryEntry<'a, K: Ord> {
    Vacant(std::collections::btree_map::VacantEntry<'a, RegistryKey<K>, Entry>),
    Occupied(&'a mut Entry),
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
        Self {
            legacy: BTreeMap::new(),
            batches: None,
        }
    }
    pub(super) fn locate(&self, key: &K) -> Option<(EntryLocator, &Entry)> {
        // Linear iteration is deliberate: the returned ordinal is consumed
        // under this same Usage loan, without another provider comparison.
        for (i, (candidate, entry)) in self.legacy.iter().enumerate() {
            let borrowed: &K = candidate.borrow();
            if borrowed.cmp(key) == Ordering::Equal {
                return Some((EntryLocator::Legacy(i), entry));
            }
        }
        self.locate_fixed(key)
    }
    fn locate_fixed(&self, key: &K) -> Option<(EntryLocator, &Entry)> {
        let mut batch = self.batches.as_deref();
        let mut index = 0;
        while let Some(current) = batch {
            for (slot, entry) in current.entries.iter().enumerate() {
                if let Some((candidate, entry)) = entry {
                    let borrowed: &K = candidate.borrow();
                    if borrowed.cmp(key) == Ordering::Equal {
                        return Some((EntryLocator::Fixed { batch: index, slot }, entry));
                    }
                }
            }
            index += 1;
            batch = current.next.as_deref();
        }
        None
    }
    pub(super) fn at_mut(&mut self, locator: EntryLocator) -> &mut Entry {
        match locator {
            EntryLocator::Legacy(i) => self
                .legacy
                .values_mut()
                .nth(i)
                .expect("validated legacy ordinal"),
            EntryLocator::Fixed { batch, slot } => {
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
        }
    }
    pub(super) fn locate_reset_layout(&self, id: u64) -> Option<(EntryLocator, &Entry)> {
        for (index, entry) in self.legacy.values().enumerate() {
            if entry.reset_layout_id == Some(id) {
                return Some((EntryLocator::Legacy(index), entry));
            }
        }
        let mut batch = self.batches.as_deref();
        let mut index = 0;
        while let Some(current) = batch {
            for (slot, entry) in current.entries.iter().enumerate() {
                if let Some((_, entry)) = entry {
                    if entry.reset_layout_id == Some(id) {
                        return Some((EntryLocator::Fixed { batch: index, slot }, entry));
                    }
                }
            }
            index += 1;
            batch = current.next.as_deref();
        }
        None
    }

    // Only runtime-issued scalar identities participate. No K::Ord, Borrow,
    // clone or destructor executes under Usage on this retirement path.
    pub(super) fn retire_reset_layout(&mut self, id: u64) -> RetiredEntry<K> {
        let (locator, entry) = self.locate_reset_layout(id).expect("live reset layout pin");
        assert_ne!(entry.owners, 0);
        if entry.owners > 1 {
            self.at_mut(locator).owners -= 1;
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
        match locator {
            EntryLocator::Legacy(_) => {
                // An unbounded extraction range never compares provider keys;
                // the predicate reads only our scalar entry identifier.
                let entry = self
                    .legacy
                    .extract_if(.., reset_layout_predicate::<K>(id))
                    .next()
                    .expect("located reset layout entry");
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
            EntryLocator::Fixed { batch, slot } => self.retire_fixed_last(batch, slot),
        }
    }

    pub(super) fn get(&self, key: &K) -> Option<&Entry> {
        self.legacy
            .get(key)
            .or_else(|| self.locate_fixed(key).map(|(_, e)| e))
    }
    fn fixed_get_mut<'a>(
        batches: &'a mut Option<Box<RegistryBatch<K>>>,
        key: &K,
    ) -> Option<&'a mut Entry> {
        let mut batch = batches.as_deref_mut();
        while let Some(current) = batch {
            for (candidate, entry) in current.entries.iter_mut().filter_map(Option::as_mut) {
                let borrowed = <RegistryKey<K> as Borrow<K>>::borrow(candidate);
                if borrowed.cmp(key) == Ordering::Equal {
                    return Some(entry);
                }
            }
            batch = current.next.as_deref_mut();
        }
        None
    }
    pub(super) fn get_mut(&mut self, key: &K) -> Option<&mut Entry> {
        match self.legacy.get_mut(key) {
            Some(entry) => Some(entry),
            None => Self::fixed_get_mut(&mut self.batches, key),
        }
    }
    pub(super) fn is_empty(&self) -> bool {
        self.legacy.is_empty() && self.batches.is_none()
    }
    pub(super) fn entry(&mut self, key: RegistryKey<K>) -> RegistryEntry<'_, K> {
        // Preserve the ordinary logarithmic map search. Only a legacy miss
        // traverses fixed nodes; a fixed match is still the same canonical entry.
        match self.legacy.entry(key) {
            std::collections::btree_map::Entry::Occupied(e) => {
                RegistryEntry::Occupied(e.into_mut())
            }
            std::collections::btree_map::Entry::Vacant(e) => {
                match Self::fixed_get_mut(&mut self.batches, e.key().borrow()) {
                    // Shared-key callers retain another Arc outside Usage;
                    // dropping this vacant key cannot drop its provider owner.
                    Some(entry) => RegistryEntry::Occupied(entry),
                    None => RegistryEntry::Vacant(e),
                }
            }
        }
    }
    pub(super) fn insert(&mut self, key: RegistryKey<K>, entry: Entry) -> Option<Entry> {
        match self.entry(key) {
            RegistryEntry::Vacant(e) => {
                e.insert(entry);
                None
            }
            RegistryEntry::Occupied(e) => Some(std::mem::replace(e, entry)),
        }
    }
    pub(super) fn link(&mut self, mut batch: Box<RegistryBatch<K>>) {
        // Prepared nonempty node; linking neither compares nor destroys keys.
        debug_assert!(batch.next.is_none());
        debug_assert!(batch.entries.iter().any(Option::is_some));
        batch.next = self.batches.take();
        self.batches = Some(batch);
    }
    pub(super) fn retire_owner(&mut self, key: &K) -> RetiredEntry<K> {
        if let Some(entry) = self.legacy.get(key) {
            assert_ne!(entry.owners, 0, "live canonical owner count");
            if entry.owners > 1 {
                self.legacy
                    .get_mut(key)
                    .expect("located legacy entry")
                    .owners -= 1;
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
            // No decrement precedes BTreeMap's final lookup/removal. Provider
            // comparison panic leaves the original owner count intact.
            let entry = self.legacy.remove_entry(key).expect("located legacy key");
            return RetiredEntry {
                entry: Some(entry),
                _batch: None,
                registration_key: None,
                namespace: None,
                _raw: None,
                _native_batch: None,
                entry_origin: None,
                _preparation: None,
            };
        }
        let (locator, entry) = self.locate_fixed(key).expect("registered storage owner");
        assert_ne!(entry.owners, 0, "live canonical owner count");
        if entry.owners > 1 {
            self.at_mut(locator).owners -= 1;
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
        match locator {
            EntryLocator::Legacy(_) => unreachable!("fixed-only lookup"),
            EntryLocator::Fixed { batch, slot } => self.retire_fixed_last(batch, slot),
        }
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

fn reset_layout_predicate<K>(id: u64) -> impl FnMut(&RegistryKey<K>, &mut Entry) -> bool {
    move |_, entry| entry.reset_layout_id == Some(id)
}

pub(super) fn reset_layout_retirement_bytes<K: Ord>() -> Option<usize> {
    // The same concrete range/predicate/iterator as retirement. An empty map
    // allocates nothing and touches no provider key or allocator-owned node.
    let mut empty = BTreeMap::<RegistryKey<K>, Entry>::new();
    let iterator = empty.extract_if(.., reset_layout_predicate::<K>(0));
    [
        std::mem::size_of_val(&iterator),
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

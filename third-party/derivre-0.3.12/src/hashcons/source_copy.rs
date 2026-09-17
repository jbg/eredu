//! Fallible copies from the actual hash-cons owner; no growth authority is granted.
use super::{hash_slice, Element, PreparedVecHashCons, VecHashCons};
use crate::RandomState;
use hashbrown::HashTable;
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    hash::BuildHasher,
    mem::{size_of, size_of_val},
};

#[derive(Debug)]
enum Cause {
    Overflow,
    Capacity,
    Vector(TryReserveError),
    Table(hashbrown::TryReserveError),
}
/// Exact actual-source copy quote. Table bytes come from the owning table's
/// existing allocation, never an estimated load factor or a caller arena size.
#[derive(Clone, Copy, Debug)]
pub struct HashConsCopyRequirements {
    words: usize,
    entries: usize,
    entry_capacity: usize,
    table_capacity: usize,
    table_bytes: usize,
    buffers: usize,
    controls: usize,
    total: usize,
}
impl HashConsCopyRequirements {
    /// All initialized backing words, including legacy padding/in-progress words.
    pub fn backing_words(&self) -> usize {
        self.words
    }
    /// Committed source vectors; their IDs are retained exactly.
    pub fn entries(&self) -> usize {
        self.entries
    }
    /// Complete destination buffers, including the source table's real allocation.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Fixed source, copier, hashing and failure representations.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Checked local copy quote; not permission for future container growth.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
/// A loan of one actual owner. No raw storage adoption or synthetic capacity input.
pub struct HashConsCopyPlan<'a> {
    source: &'a VecHashCons,
    requirements: HashConsCopyRequirements,
}
impl fmt::Debug for HashConsCopyPlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashConsCopyPlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
/// A failed copy retains every actual destination allocation until this owner
/// retires. The enclosing admission owner must retain its funding alongside it.
pub struct HashConsCopyFailure {
    cause: Cause,
    copy: Option<VecHashCons>,
}
impl fmt::Debug for HashConsCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashConsCopyFailure")
            .field("cause", &self.cause)
            .field("retains_prefix", &self.copy.is_some())
            .finish()
    }
}
impl fmt::Display for HashConsCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Overflow => f.write_str("hash-cons copy geometry overflow"),
            Cause::Capacity => {
                f.write_str("hash-cons copy destination differs from its source extent")
            }
            Cause::Vector(e) => fmt::Display::fmt(e, f),
            Cause::Table(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for HashConsCopyFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Vector(e) => Some(e),
            Cause::Table(e) => Some(e),
            _ => None,
        }
    }
}
impl VecHashCons {
    /// Quotes copies of the same vectors, hasher, table entries and pending
    /// legacy insertion state. A pending source is not certified as quiescent.
    pub fn source_copy_plan(&self) -> Result<HashConsCopyPlan<'_>, HashConsCopyFailure> {
        HashConsCopyPlan::prepare(self).map_err(|cause| HashConsCopyFailure { cause, copy: None })
    }

    /// Prepares a finite independent table using this quiescent source's actual
    /// initialized backing and table capacity. The largest source vector supplies
    /// one additional encoding-scratch destination. No guessed growth factor or
    /// caller capacity grants additional storage.
    ///
    /// The result remains a raw table: copied IDs do not certify expression
    /// children, predefined seeds, or an ExprSet/lexer/parser execution source.
    pub fn prepared_source_plan(
        &self,
    ) -> Result<HashConsPreparedSourcePlan<'_>, HashConsCopyFailure> {
        HashConsPreparedSourcePlan::prepare(self)
            .map_err(|cause| HashConsCopyFailure { cause, copy: None })
    }
}

impl PreparedVecHashCons {
    /// Copies the same finite limits, including the distinction between committed
    /// payload and encoding scratch. A pending insertion cannot supply a source.
    pub fn prepared_source_plan(
        &self,
    ) -> Result<HashConsPreparedSourcePlan<'_>, HashConsCopyFailure> {
        HashConsPreparedSourcePlan::from_prepared(self)
            .map_err(|cause| HashConsCopyFailure { cause, copy: None })
    }
}

/// Actual source-derived limits and complete destination construction quote.
#[derive(Clone, Copy, Debug)]
pub struct HashConsPreparedSourceRequirements {
    copy: HashConsCopyRequirements,
    max_words: usize,
    max_entries: usize,
    max_encoded_words: usize,
}
impl HashConsPreparedSourceRequirements {
    /// Maximum committed payload words, from the source's initialized backing.
    pub fn max_words(&self) -> usize {
        self.max_words
    }
    /// Maximum unique entries, from the source table's actual usable capacity.
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }
    /// Largest actual source vector; reserved again as independent scratch.
    pub fn max_encoded_words(&self) -> usize {
        self.max_encoded_words
    }
    /// All independently constructed vectors and the actual source table layout.
    pub fn buffer_bytes(&self) -> usize {
        self.copy.buffers
    }
    /// Local source, copying, and completed-destination constructor frames.
    pub fn control_bytes(&self) -> usize {
        self.copy.controls
    }
    /// Complete local constructor requirement; not expression execution authority.
    pub fn required_bytes(&self) -> usize {
        self.copy.total
    }
}

/// Exclusive source geometry for a separately paid finite append destination.
/// The immutable source is only borrowed; no raw storage is adopted.
pub struct HashConsPreparedSourcePlan<'a> {
    copy: HashConsCopyPlan<'a>,
    requirements: HashConsPreparedSourceRequirements,
}
impl fmt::Debug for HashConsPreparedSourcePlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashConsPreparedSourcePlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
impl<'a> HashConsPreparedSourcePlan<'a> {
    fn wrapper_controls() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<HashConsPreparedSourceRequirements>(),
            size_of::<PreparedVecHashCons>(),
            size_of::<Result<Self, Cause>>(),
            size_of::<Result<Self, HashConsCopyFailure>>(),
            size_of::<Result<PreparedVecHashCons, HashConsCopyFailure>>(),
            size_of::<std::slice::Iter<'_, Element>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    fn from_prepared(source: &'a PreparedVecHashCons) -> Result<Self, Cause> {
        if source.insertion_active
            || source.inner.curr_elt.backing_end != 0
            || source.inner.curr_elt.backing_start < 4
            || source.inner.table.len() != source.inner.elements.len()
        {
            return Err(Cause::Capacity);
        }
        let words = source
            .max_words
            .checked_add(4)
            .and_then(|n| n.checked_add(source.max_encoded_words))
            .ok_or(Cause::Overflow)?;
        if words != source.inner.backing.len()
            || source.inner.elements.len() > source.max_entries
            || source.max_entries > source.inner.table.capacity()
            || source.inner.curr_elt.backing_start as usize - 4 > source.max_words
        {
            return Err(Cause::Capacity);
        }
        let mut copy = HashConsCopyPlan::prepare(&source.inner)?;
        let requirements = &mut copy.requirements;
        requirements.entry_capacity = source.max_entries;
        requirements.buffers = Layout::array::<u32>(words)
            .map_err(|_| Cause::Overflow)?
            .size()
            .checked_add(
                Layout::array::<Element>(source.max_entries)
                    .map_err(|_| Cause::Overflow)?
                    .size(),
            )
            .and_then(|n| n.checked_add(requirements.table_bytes))
            .ok_or(Cause::Overflow)?;
        requirements.controls = Self::wrapper_controls()
            .and_then(|n| n.checked_add(requirements.controls))
            .and_then(|n| n.checked_add(size_of::<&PreparedVecHashCons>()))
            .and_then(|n| n.checked_add(size_of::<Result<Self, Cause>>()))
            .ok_or(Cause::Overflow)?;
        requirements.total = requirements
            .buffers
            .checked_add(requirements.controls)
            .ok_or(Cause::Overflow)?;
        let requirements = HashConsPreparedSourceRequirements {
            copy: *requirements,
            max_words: source.max_words,
            max_entries: source.max_entries,
            max_encoded_words: source.max_encoded_words,
        };
        Ok(Self { copy, requirements })
    }
    fn prepare(source: &'a VecHashCons) -> Result<Self, Cause> {
        if source.curr_elt.backing_end != 0
            || source.curr_elt.backing_start < 4
            || source.curr_elt.backing_start as usize > source.backing.len()
            || source.table.len() != source.elements.len()
        {
            return Err(Cause::Capacity);
        }
        let max_words = source.backing.len().checked_sub(4).ok_or(Cause::Overflow)?;
        let max_entries = source.table.capacity();
        let mut max_encoded_words = 0;
        for entry in &source.elements {
            if entry.backing_start < 4
                || entry.backing_start > entry.backing_end
                || entry.backing_end > source.curr_elt.backing_start
            {
                return Err(Cause::Capacity);
            }
            max_encoded_words =
                max_encoded_words.max((entry.backing_end - entry.backing_start) as usize);
        }
        let words = source
            .backing
            .len()
            .checked_add(max_encoded_words)
            .ok_or(Cause::Overflow)?;
        u32::try_from(words).map_err(|_| Cause::Overflow)?;
        u32::try_from(max_entries).map_err(|_| Cause::Overflow)?;
        let mut copy = HashConsCopyPlan::prepare(source)?;
        copy.requirements.words = words;
        copy.requirements.entry_capacity = max_entries;
        copy.requirements.buffers = Layout::array::<u32>(words)
            .map_err(|_| Cause::Overflow)?
            .size()
            .checked_add(
                Layout::array::<Element>(max_entries)
                    .map_err(|_| Cause::Overflow)?
                    .size(),
            )
            .and_then(|n| n.checked_add(copy.requirements.table_bytes))
            .ok_or(Cause::Overflow)?;
        copy.requirements.controls = Self::wrapper_controls()
            .and_then(|n| n.checked_add(copy.requirements.controls))
            .ok_or(Cause::Overflow)?;
        copy.requirements.total = copy
            .requirements
            .buffers
            .checked_add(copy.requirements.controls)
            .ok_or(Cause::Overflow)?;
        let requirements = HashConsPreparedSourceRequirements {
            copy: copy.requirements,
            max_words,
            max_entries,
            max_encoded_words,
        };
        Ok(Self { copy, requirements })
    }
    /// Complete local requirements for this same source and real table layout.
    pub fn requirements(&self) -> HashConsPreparedSourceRequirements {
        self.requirements
    }
    /// Uses the ordinary source-copy worker, with separately priced physical
    /// element slots and encoding scratch. All partial copies remain in the
    /// returned failure; successful insertion uses the existing fixed worker.
    pub fn compile(self) -> Result<PreparedVecHashCons, HashConsCopyFailure> {
        let inner = self.copy.compile()?;
        Ok(PreparedVecHashCons {
            inner,
            max_words: self.requirements.max_words,
            max_entries: self.requirements.max_entries,
            max_encoded_words: self.requirements.max_encoded_words,
            insertion_active: false,
            backing_funding: None,
        })
    }
}
/// Empty fixed intern table with the exact backing and table layout of a
/// retained source. Encoding scratch is separately quoted from the caller's
/// actual operation shape. This supplies storage, not expression or DFA IDs.
pub struct HashConsEmptySourcePlan<'a> {
    plan: HashConsPreparedSourcePlan<'a>,
}
impl fmt::Debug for HashConsEmptySourcePlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashConsEmptySourcePlan")
            .field("requirements", &self.plan.requirements)
            .finish()
    }
}
impl<'a> HashConsPreparedSourcePlan<'a> {
    /// Complete fixed inspection/copy frames before source traversal.
    pub fn inspection_control_bytes() -> Option<usize> {
        Self::wrapper_controls()?
            .checked_add(HashConsCopyPlan::control_bytes()?)
            .and_then(|n| n.checked_add(size_of::<&PreparedVecHashCons>()))
            .and_then(|n| n.checked_add(size_of::<Result<Self, Cause>>()))
    }
    /// Reuses this source's actual committed-payload and hash-table extents for
    /// an independent initially empty destination, with exact encoding scratch.
    pub fn empty_destination(
        mut self,
        encoding_words: usize,
    ) -> Result<HashConsEmptySourcePlan<'a>, HashConsCopyFailure> {
        let result = (|| -> Result<(), Cause> {
            let words = self
                .requirements
                .max_words
                .checked_add(4)
                .and_then(|n| n.checked_add(encoding_words))
                .ok_or(Cause::Overflow)?;
            u32::try_from(words).map_err(|_| Cause::Overflow)?;
            let old = Layout::array::<u32>(self.copy.requirements.words)
                .map_err(|_| Cause::Overflow)?
                .size();
            let new = Layout::array::<u32>(words)
                .map_err(|_| Cause::Overflow)?
                .size();
            self.copy.requirements.buffers = self
                .copy
                .requirements
                .buffers
                .checked_sub(old)
                .and_then(|n| n.checked_add(new))
                .ok_or(Cause::Overflow)?;
            self.copy.requirements.controls = HashConsEmptySourcePlan::wrapper_controls()
                .and_then(|n| n.checked_add(self.copy.requirements.controls))
                .ok_or(Cause::Overflow)?;
            self.copy.requirements.total = self
                .copy
                .requirements
                .buffers
                .checked_add(self.copy.requirements.controls)
                .ok_or(Cause::Overflow)?;
            self.copy.requirements.words = words;
            self.requirements.copy = self.copy.requirements;
            self.requirements.max_encoded_words = encoding_words;
            Ok(())
        })();
        result.map_err(|cause| HashConsCopyFailure { cause, copy: None })?;
        Ok(HashConsEmptySourcePlan { plan: self })
    }
}
impl HashConsEmptySourcePlan<'_> {
    fn wrapper_controls() -> Option<usize> {
        let controls = [
            size_of::<HashConsEmptySourcePlan<'_>>(),
            size_of::<usize>() * 4,
            size_of::<Result<HashConsEmptySourcePlan<'_>, HashConsCopyFailure>>(),
            size_of::<Result<(), Cause>>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
    }
    /// Fixed source-layout and empty-destination inspection frames, before the
    /// source is traversed and before a destination quote is assembled.
    pub fn inspection_control_bytes() -> Option<usize> {
        Self::wrapper_controls()?
            .checked_add(HashConsPreparedSourcePlan::inspection_control_bytes()?)
    }

    /// Complete exact storage quote for this same source loan.
    pub fn requirements(&self) -> HashConsPreparedSourceRequirements {
        self.plan.requirements
    }
    /// Constructs through the shared copy/destination worker, without copying
    /// source IDs or exposing mutable ordinary storage.
    pub fn compile(self) -> Result<PreparedVecHashCons, HashConsCopyFailure> {
        let inner = self.plan.copy.compile_with(true)?;
        Ok(PreparedVecHashCons {
            inner,
            max_words: self.plan.requirements.max_words,
            max_entries: self.plan.requirements.max_entries,
            max_encoded_words: self.plan.requirements.max_encoded_words,
            insertion_active: false,
            backing_funding: None,
        })
    }
}
impl<'a> HashConsCopyPlan<'a> {
    pub(crate) fn control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<HashConsCopyRequirements>(),
            size_of::<HashConsCopyFailure>(),
            size_of::<Cause>(),
            size_of::<VecHashCons>(),
            size_of::<Option<VecHashCons>>(),
            size_of::<HashTable<u32>>(),
            size_of::<RandomState>(),
            size_of::<<RandomState as BuildHasher>::Hasher>(),
            size_of::<Vec<u32>>(),
            size_of::<Vec<Element>>(),
            size_of::<Element>(),
            size_of::<Result<Self, Cause>>(),
            size_of::<Result<Self, HashConsCopyFailure>>(),
            size_of::<Result<VecHashCons, HashConsCopyFailure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<(), hashbrown::TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<hashbrown::hash_table::Iter<'_, u32>>(),
            size_of::<hashbrown::hash_table::OccupiedEntry<'_, u32>>(),
            size_of::<std::slice::Iter<'_, u32>>(),
            size_of::<std::slice::Iter<'_, Element>>(),
            size_of::<(&VecHashCons, u32)>(),
            size_of::<(&RandomState, &[u32])>(),
            size_of::<(&VecHashCons,)>(),
            size_of::<(&mut VecHashCons,)>(),
            size_of::<bool>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    fn prepare(source: &'a VecHashCons) -> Result<Self, Cause> {
        let words = source.backing.len();
        let entries = source.elements.len();
        let table_capacity = source.table.capacity();
        let table_bytes = source.table.allocation_size();
        let buffers = Layout::array::<u32>(words)
            .map_err(|_| Cause::Overflow)?
            .size()
            .checked_add(
                Layout::array::<Element>(entries)
                    .map_err(|_| Cause::Overflow)?
                    .size(),
            )
            .and_then(|n| n.checked_add(table_bytes))
            .ok_or(Cause::Overflow)?;
        let controls = Self::control_bytes().ok_or(Cause::Overflow)?;
        let total = buffers.checked_add(controls).ok_or(Cause::Overflow)?;
        Ok(Self {
            source,
            requirements: HashConsCopyRequirements {
                words,
                entries,
                entry_capacity: entries,
                table_capacity,
                table_bytes,
                buffers,
                controls,
                total,
            },
        })
    }
    /// Complete local requirements for this same source loan.
    pub fn requirements(&self) -> HashConsCopyRequirements {
        self.requirements
    }
    /// Constructs each vector/table once and verifies actual capacities before
    /// copying entries. Ordinary Clone uses this same worker. Future mutation
    /// remains governed by the caller's separate growth/admission contract.
    pub fn compile(self) -> Result<VecHashCons, HashConsCopyFailure> {
        self.compile_with(false)
    }
    fn compile_with(self, empty: bool) -> Result<VecHashCons, HashConsCopyFailure> {
        let source = self.source;
        let mut copy = VecHashCons {
            hasher: source.hasher.clone(),
            backing: Vec::new(),
            elements: Vec::new(),
            table: HashTable::new(),
            curr_elt: if empty {
                Element {
                    backing_start: 4,
                    backing_end: 0,
                }
            } else {
                source.curr_elt.clone()
            },
        };
        let result = (|| -> Result<(), Cause> {
            copy.backing
                .try_reserve_exact(self.requirements.words)
                .map_err(Cause::Vector)?;
            if copy.backing.capacity() != self.requirements.words {
                return Err(Cause::Capacity);
            }
            if !empty {
                copy.backing.extend_from_slice(&source.backing);
            }
            copy.backing.resize(self.requirements.words, 0);
            copy.elements
                .try_reserve_exact(self.requirements.entry_capacity)
                .map_err(Cause::Vector)?;
            if copy.elements.capacity() != self.requirements.entry_capacity {
                return Err(Cause::Capacity);
            }
            if !empty {
                copy.elements.extend_from_slice(&source.elements);
            }
            // Starting from an empty table, reserve the actual source table's
            // usable capacity. The owning implementation selects its layout;
            // verify it against that same source before any insertion.
            copy.table
                .try_reserve(self.requirements.table_capacity, |_| 0)
                .map_err(Cause::Table)?;
            if copy.table.capacity() != self.requirements.table_capacity
                || copy.table.allocation_size() != self.requirements.table_bytes
            {
                return Err(Cause::Capacity);
            }
            if !empty {
                for &id in source.table.iter() {
                    if copy.table.len() == copy.table.capacity() {
                        return Err(Cause::Capacity);
                    }
                    let hash = hash_slice(&source.hasher, source.get(id));
                    copy.table
                        .insert_unique(hash, id, |id| hash_slice(&source.hasher, source.get(*id)));
                }
                if copy.table.len() != source.table.len() {
                    return Err(Cause::Capacity);
                }
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(copy),
            Err(cause) => Err(HashConsCopyFailure {
                cause,
                copy: Some(copy),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_template_preserves_ids_and_fixed_append_failure_custody() {
        let mut source = VecHashCons::new();
        let empty = source.insert(&[]);
        let first = source.insert(&[7, 11, 13]);
        let source_entries = source.len();
        let source_words = source.backing.len();
        let source_capacity = source.table.capacity();
        let source_bytes = source.retained_capacity_bytes().unwrap();
        let plan = source.prepared_source_plan().unwrap();
        let requirements = plan.requirements();
        assert_eq!(requirements.max_words(), source_words - 4);
        assert_eq!(requirements.max_entries(), source_capacity);
        assert_eq!(requirements.max_encoded_words(), 3);
        assert_eq!(
            requirements.required_bytes(),
            requirements.buffer_bytes() + requirements.control_bytes()
        );
        let mut copy = plan.compile().unwrap();
        assert_eq!(copy.lookup(&[]), Some(empty));
        assert_eq!(copy.lookup(&[7, 11, 13]), Some(first));
        assert_eq!(
            copy.retained_capacity_bytes().unwrap(),
            requirements.buffer_bytes()
        );
        let bytes = copy.retained_capacity_bytes().unwrap();
        for i in source_entries..source_capacity {
            assert_eq!(copy.try_insert(&[100 + i as u32]).unwrap(), i as u32);
        }
        assert!(matches!(
            copy.try_insert(&[999]),
            Err(super::super::HashConsCapacityError::EntriesExceeded { .. })
        ));
        let mut duplicate = copy.begin_insert(3).unwrap();
        duplicate.push_slice(&[7, 11, 13]).unwrap();
        assert_eq!(duplicate.finish().unwrap(), first);
        assert_eq!(copy.retained_capacity_bytes().unwrap(), bytes);
        assert_eq!(source.len(), source_entries);
        assert_eq!(source.retained_capacity_bytes().unwrap(), source_bytes);
        assert_eq!(source.lookup(&[100 + source_entries as u32]), None);

        let copied_plan = copy.prepared_source_plan().unwrap();
        let copied_requirements = copied_plan.requirements();
        assert_eq!(copied_requirements.max_words(), copy.max_words());
        assert_eq!(copied_requirements.max_entries(), copy.max_entries());
        assert_eq!(
            copied_requirements.max_encoded_words(),
            copy.max_encoded_words()
        );
        assert_eq!(copied_requirements.buffer_bytes(), bytes);
        let mut copied_again = copied_plan.compile().unwrap();
        assert_eq!(copied_again.lookup(&[7, 11, 13]), Some(first));
        assert!(matches!(
            copied_again.try_insert(&[999]),
            Err(super::super::HashConsCapacityError::EntriesExceeded { .. })
        ));
        drop(copied_again);

        let mut refused = source.prepared_source_plan().unwrap();
        refused.copy.requirements.table_bytes = 0;
        let failure = match refused.compile() {
            Err(error) => error,
            Ok(_) => panic!("different table layout was accepted"),
        };
        assert!(matches!(failure.cause, Cause::Capacity));
        source.start_insert();
        source.push_u32(29);
        let pending = source.prepared_source_plan().unwrap_err();
        assert!(matches!(pending.cause, Cause::Capacity));
        assert!(pending.copy.is_none());
        drop(source);
        drop(copy);
        let prefix = failure.copy.as_ref().unwrap();
        assert_eq!(prefix.backing.len(), source_words + 3);
        assert_eq!(prefix.elements.len(), source_entries);
        assert_eq!(prefix.elements.capacity(), source_capacity);
        assert!(prefix.table.allocation_size() > 0);
        assert!(prefix.table.is_empty());
    }

    #[test]
    fn source_copy_preserves_ids_pending_words_and_failed_destination_custody() {
        let empty = VecHashCons::new();
        let empty_copy = empty.source_copy_plan().unwrap().compile().unwrap();
        assert_eq!(empty_copy.curr_elt.backing_start, 4);
        assert_eq!(empty_copy.curr_elt.backing_end, 0);
        assert_eq!(empty_copy.retained_capacity_bytes().unwrap(), 0);

        let mut source = VecHashCons::new();
        source.reserve(50);
        let empty_id = source.insert(&[]);
        let first = source.insert(&[7, 11, 13]);
        let second = source.insert(&[19, 23]);
        assert_eq!(source.insert(&[7, 11, 13]), first);
        source.start_insert();
        source.push_slice(&[29, 31]);
        let plan = source.source_copy_plan().unwrap();
        let requirements = plan.requirements();
        let mut copied = plan.compile().unwrap();
        let mut ordinary = source.clone();
        assert_eq!(
            copied.retained_capacity_bytes().unwrap(),
            requirements.buffer_bytes()
        );
        assert_eq!(copied.backing.capacity(), source.backing.len());
        assert_eq!(copied.elements.capacity(), source.elements.len());
        assert_eq!(copied.table.capacity(), source.table.capacity());
        assert_eq!(copied.backing, source.backing);
        assert_eq!(
            hash_slice(&copied.hasher, &[5, 17]),
            hash_slice(&source.hasher, &[5, 17])
        );
        assert_eq!(copied.lookup(&[]), Some(empty_id));
        assert_eq!(copied.lookup(&[7, 11, 13]), Some(first));
        assert_eq!(copied.lookup(&[19, 23]), Some(second));
        assert_eq!(copied.lookup(&[29, 31]), None);
        copied.push_u32(37);
        ordinary.push_u32(37);
        assert_eq!(copied.finish_insert(), ordinary.finish_insert());
        assert_eq!(source.lookup(&[29, 31, 37]), None);
        source.push_u32(37);
        assert_eq!(
            source.finish_insert(),
            copied.lookup(&[29, 31, 37]).unwrap()
        );
        assert_eq!(source.insert(&[7, 11, 13]), first);

        // A late table-layout rejection owns the already copied word/entry
        // vectors and actual newly allocated table, independently of source.
        let mut refused = source.source_copy_plan().unwrap();
        refused.requirements.table_bytes = 0;
        let failure = match refused.compile() {
            Err(failure) => failure,
            Ok(_) => panic!("mismatched layout was accepted"),
        };
        assert!(matches!(failure.cause, Cause::Capacity));
        let expected_words = source.backing.clone();
        let expected_entries = source.elements.len();
        assert_eq!(source.lookup(&[29, 31, 37]), copied.lookup(&[29, 31, 37]));
        drop(source);
        drop(copied);
        drop(ordinary);
        let prefix = failure.copy.as_ref().unwrap();
        assert_eq!(prefix.backing, expected_words);
        assert_eq!(prefix.elements.len(), expected_entries);
        assert!(prefix.table.allocation_size() > 0);
        assert!(prefix.table.is_empty());
    }
}

mod growth;
mod source_copy;
pub use growth::{
    HashConsFundingFailure, HashConsFundingPreparationError, PreparedHashConsFunding,
};
use hashbrown::hash_table::Entry;
use std::{
    collections::TryReserveError,
    fmt,
    hash::{BuildHasher, Hasher},
};

use hashbrown::HashTable;

use crate::RandomState;

#[derive(Debug, Clone)]
struct Element {
    backing_start: u32,
    backing_end: u32,
}

impl Element {
    fn as_range(&self) -> std::ops::Range<usize> {
        (self.backing_start as usize)..(self.backing_end as usize)
    }
}

/// A hashconsing data structure for vectors of u32.
/// Given a vector, it stores it only once and returns a unique id.
/// The ids are consecutive and start at 0.
pub struct VecHashCons {
    hasher: RandomState,
    backing: Vec<u32>,
    elements: Vec<Element>,
    table: HashTable<u32>,
    curr_elt: Element,
}

impl Clone for VecHashCons {
    fn clone(&self) -> Self {
        self.source_copy_plan()
            .expect("ordinary hash-cons copy geometry")
            .compile()
            .expect("ordinary hash-cons copy allocation")
    }
}

impl Default for VecHashCons {
    fn default() -> Self {
        Self::new()
    }
}

impl VecHashCons {
    /// Create a new hashcons.
    pub fn new() -> Self {
        VecHashCons {
            hasher: RandomState::new(),
            backing: Vec::new(),
            elements: Vec::new(),
            table: HashTable::new(),
            // we start at 4, so there is no implied initial start_insert()
            // but data is still aligned if needed
            curr_elt: Element {
                backing_start: 4,
                backing_end: 0,
            },
        }
    }

    /// Insert a given vector and return its unique id.
    pub fn insert(&mut self, data: &[u32]) -> u32 {
        self.start_insert();
        self.push_slice(data);
        self.finish_insert()
    }

    /// Finds an already interned vector without allocating, inserting, or
    /// changing incremental-insertion scratch. This also works while a legacy
    /// incremental insertion is in progress.
    pub fn lookup(&self, data: &[u32]) -> Option<u32> {
        let hash = hash_slice(&self.hasher, data);
        self.table.find(hash, |id| self.get(*id) == data).copied()
    }

    /// Checks representation and requested vector layout before legacy insertion.
    /// Call [`Self::lookup`] first if a duplicate need not consume another entry.
    /// This does not reserve storage, validate a byte allowance, or make growing
    /// insertion transactional. The legacy allocation path remains unrestricted.
    pub fn validate_insert_geometry(&self, words: usize) -> Result<(), HashConsCapacityError> {
        if self.curr_elt.backing_end != 0 {
            return Err(HashConsCapacityError::InsertionInProgress);
        }
        let words = u32::try_from(words).map_err(|_| HashConsCapacityError::IndexOverflow)?;
        let end = self
            .curr_elt
            .backing_start
            .checked_add(words)
            .ok_or(HashConsCapacityError::IndexOverflow)? as usize;
        if self.backing.len() < end {
            // Match ensure_size's explicit padding without relying on the
            // allocator's eventual capacity or authorizing that allocation.
            let requested = end
                .checked_add(128)
                .ok_or(HashConsCapacityError::CapacityOverflow)?;
            if allocation_bytes::<u32>(requested)? > isize::MAX as usize {
                return Err(HashConsCapacityError::CapacityOverflow);
            }
        }
        let entries = self
            .elements
            .len()
            .checked_add(1)
            .ok_or(HashConsCapacityError::CapacityOverflow)?;
        u32::try_from(entries).map_err(|_| HashConsCapacityError::IndexOverflow)?;
        if allocation_bytes::<Element>(entries)? > isize::MAX as usize {
            return Err(HashConsCapacityError::CapacityOverflow);
        }
        Ok(())
    }

    /// Exact retained allocation capacity of this hash-consing container.
    ///
    /// Includes spare vector capacity and the hash table's complete allocation,
    /// using the owning table implementation's layout. It excludes this inline
    /// Rust value and allocator bookkeeping outside the requested allocations.
    /// This is cold inventory, not a bound for future growth or for ExprSet,
    /// lexer, parser, or clone storage.
    pub fn retained_capacity_bytes(&self) -> Result<usize, HashConsCapacityError> {
        let backing = allocation_bytes::<u32>(self.backing.capacity())?;
        let elements = allocation_bytes::<Element>(self.elements.capacity())?;
        backing
            .checked_add(elements)
            .and_then(|bytes| bytes.checked_add(self.table.allocation_size()))
            .ok_or(HashConsCapacityError::CapacityOverflow)
    }

    /// Get vector with given unique id.
    /// Panics if id is out of bounds.
    #[inline(always)]
    pub fn get(&self, id: u32) -> &[u32] {
        &self.backing[self.elements[id as usize].as_range()]
    }

    pub fn is_valid(&self, id: u32) -> bool {
        id < self.elements.len() as u32
    }

    /// Return number of elements in the hashcons (also largest unique id + 1).
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Estimate number of bytes used by the hashcons.
    pub fn num_bytes(&self) -> usize {
        self.backing.len() * std::mem::size_of::<u32>()
            + self.elements.len() * (5 + std::mem::size_of::<Element>())
    }

    // Incremental, zero-copy insertion:

    /// Start insertion process for a vector.
    /// Panics if start_insert() is called twice without finish_insert().
    #[inline(always)]
    pub fn start_insert(&mut self) {
        assert!(self.curr_elt.backing_end == 0);
        self.curr_elt.backing_end = self.curr_elt.backing_start;
    }

    #[inline(always)]
    fn ensure_size(&mut self) {
        assert!(self.curr_elt.backing_end >= self.curr_elt.backing_start);
        let size = self.curr_elt.backing_end as usize;
        if self.backing.len() < size {
            self.initialize_backing_with(size + 128, |backing, total| {
                backing.reserve(total - backing.len());
                Ok::<(), std::convert::Infallible>(())
            })
            .unwrap();
        }
    }

    /// Add an element to the vector being inserted.
    /// Requires start_insert() to have been called.
    #[inline(always)]
    pub fn push_u32(&mut self, head: u32) {
        self.curr_elt.backing_end += 1;
        self.ensure_size();
        self.backing[self.curr_elt.backing_end as usize - 1] = head;
    }

    /// Add a slice to the vector being inserted.
    /// Requires start_insert() to have been called.
    #[inline(always)]
    pub fn push_slice(&mut self, elts: &[u32]) {
        let slice_start = self.curr_elt.backing_end;
        self.curr_elt.backing_end += elts.len() as u32;
        self.ensure_size();
        self.backing[slice_start as usize..self.curr_elt.backing_end as usize]
            .copy_from_slice(elts);
    }

    pub fn reserve(&mut self, size: usize) {
        self.backing.reserve(size * 4);
        self.elements.reserve(size);

        let hash_slice = |x: &[u32]| -> u64 {
            let mut hasher = self.hasher.build_hasher();
            hasher.write(bytemuck::cast_slice(x));
            hasher.finish()
        };
        let get_slice =
            |x: &u32| -> &[u32] { &self.backing[self.elements[*x as usize].as_range()] };
        let hasher = |x: &u32| -> u64 { hash_slice(get_slice(x)) };

        self.table.reserve(size, hasher);
    }

    /// Finish insertion process for a vector.
    /// Returns the unique id of the vector.
    /// Requires start_insert() to have been called.
    pub fn finish_insert(&mut self) -> u32 {
        let hash_slice = |x: &[u32]| -> u64 {
            let mut hasher = self.hasher.build_hasher();
            hasher.write(bytemuck::cast_slice(x));
            // x.hash(&mut hasher);
            hasher.finish()
        };
        let curr_backing = &self.backing[self.curr_elt.as_range()];
        let hash = hash_slice(curr_backing);
        let get_slice =
            |x: &u32| -> &[u32] { &self.backing[self.elements[*x as usize].as_range()] };
        let hasher = |x: &u32| -> u64 { hash_slice(get_slice(x)) };
        let eq = |x: &u32| -> bool { get_slice(x) == curr_backing };

        match self.table.entry(hash, eq, hasher) {
            Entry::Occupied(e) => {
                self.curr_elt.backing_end = 0;
                *e.get()
            }
            Entry::Vacant(e) => {
                let id = self.elements.len() as u32;
                self.elements.push(self.curr_elt.clone());
                e.insert(id);
                self.curr_elt.backing_start = self.curr_elt.backing_end;
                self.curr_elt.backing_end = 0;
                id
            }
        }
    }
}

fn hash_slice(hasher: &RandomState, data: &[u32]) -> u64 {
    let mut hash = hasher.build_hasher();
    hash.write(bytemuck::cast_slice(data));
    hash.finish()
}

fn allocation_bytes<T>(elements: usize) -> Result<usize, HashConsCapacityError> {
    elements
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(HashConsCapacityError::CapacityOverflow)
}

/// A checked hash-consing storage operation could not proceed.
#[derive(Debug)]
pub enum HashConsCapacityError {
    /// Element counts or complete retained byte totals overflowed usize.
    CapacityOverflow,
    /// A backing offset or vector id cannot be represented by this container.
    IndexOverflow,
    /// A legacy incremental insertion must finish before another can begin.
    InsertionInProgress,
    /// Allocating vector storage during preparation failed.
    VectorAllocation(TryReserveError),
    /// Allocating hash-table storage during preparation failed.
    TableAllocation(hashbrown::TryReserveError),
    /// The actual retained backing-growth account refused the next allocation.
    Funding(HashConsFundingFailure),
    /// A new vector would exceed the prepared payload-word allowance.
    WordsExceeded {
        required_words: usize,
        capacity_words: usize,
    },
    /// A new unique vector would exceed the prepared entry allowance.
    EntriesExceeded {
        required_entries: usize,
        capacity_entries: usize,
    },
    /// An encoding exceeds the separately prepared staging-word allowance.
    ScratchExceeded {
        required_words: usize,
        capacity_words: usize,
    },
    /// An incremental encoding did not write exactly its declared word count.
    EncodedLengthMismatch {
        expected_words: usize,
        actual_words: usize,
    },
    /// The prepared storage no longer satisfies its private invariants.
    InvalidPreparedStorage,
}

impl fmt::Display for HashConsCapacityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapacityOverflow => formatter.write_str("hash-consing capacity overflow"),
            Self::IndexOverflow => {
                formatter.write_str("hash-consing index exceeds u32 representation")
            }
            Self::InsertionInProgress => {
                formatter.write_str("hash-consing insertion already in progress")
            }
            Self::VectorAllocation(error) => {
                write!(formatter, "hash-consing vector preparation failed: {error}")
            }
            Self::TableAllocation(error) => {
                write!(formatter, "hash-consing table preparation failed: {error}")
            }
            Self::Funding(error)=>fmt::Display::fmt(error,formatter),
            Self::WordsExceeded {
                required_words,
                capacity_words,
            } => write!(
                formatter,
                "hash-consing requires {required_words} words but prepared capacity is {capacity_words}"
            ),
            Self::EntriesExceeded {
                required_entries,
                capacity_entries,
            } => write!(
                formatter,
                "hash-consing requires {required_entries} entries but prepared capacity is {capacity_entries}"
            ),
            Self::ScratchExceeded {
                required_words,
                capacity_words,
            } => write!(
                formatter,
                "hash-consing encoding requires {required_words} scratch words but prepared capacity is {capacity_words}"
            ),
            Self::EncodedLengthMismatch {
                expected_words,
                actual_words,
            } => write!(
                formatter,
                "hash-consing encoding declared {expected_words} words but wrote or attempted {actual_words}"
            ),
            Self::InvalidPreparedStorage => {
                formatter.write_str("hash-consing prepared storage is inconsistent")
            }
        }
    }
}

impl std::error::Error for HashConsCapacityError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::VectorAllocation(error) => Some(error),
            Self::TableAllocation(error) => Some(error),
            Self::Funding(error) => Some(error),
            _ => None,
        }
    }
}

/// An intern table with fixed insertion storage by default. An explicitly bound
/// paid storage owner can reserve reached payload/scratch, entry-vector and
/// source-quoted hash-table growth before publication.
///
/// Preparation allocates under the caller's existing authority. Its requested
/// word/entry limits are logical limits, not a byte admission proof: actual
/// vector and table capacity can exceed those requests. Inspect
/// [`Self::retained_capacity_bytes`] afterward for exact retained storage.
/// This owner exposes no mutable legacy container or implicit allocating clone.
/// It certifies only its own insertion mechanism, not an enclosing parser.
pub struct PreparedVecHashCons {
    inner: VecHashCons,
    max_words: usize,
    max_entries: usize,
    max_encoded_words: usize,
    insertion_active: bool,
    backing_funding: Option<PreparedHashConsFunding>,
}

impl PreparedVecHashCons {
    pub(crate) fn source(&self) -> &VecHashCons {
        &self.inner
    }

    /// Fallibly allocates all storage before any bounded insertion. The word
    /// allowance covers unique payload words; the legacy four-word prefix is
    /// additional retained backing. Empty vectors consume an entry but no words.
    pub fn try_new(max_words: usize, max_entries: usize) -> Result<Self, HashConsCapacityError> {
        Self::try_new_with_scratch(max_words, max_entries, 0)
    }

    /// Prepares an additional staging allowance for one incremental encoding.
    /// The initialized backing covers the four-word prefix, all committed
    /// payload words, and `max_encoded_words` more words. Thus a duplicate can
    /// still be encoded when the committed word and entry allowances are full.
    ///
    /// This allocates under the caller's existing authority. Requested limits
    /// do not certify byte admission; actual retained capacities may be larger.
    pub fn try_new_with_scratch(
        max_words: usize,
        max_entries: usize,
        max_encoded_words: usize,
    ) -> Result<Self, HashConsCapacityError> {
        let backing_words = max_words
            .checked_add(4)
            .and_then(|words| words.checked_add(max_encoded_words))
            .ok_or(HashConsCapacityError::CapacityOverflow)?;
        u32::try_from(backing_words).map_err(|_| HashConsCapacityError::IndexOverflow)?;
        // Keeping the length representable also preserves legacy is_valid's
        // comparison, rather than permitting a wrapping u32 element count.
        u32::try_from(max_entries).map_err(|_| HashConsCapacityError::IndexOverflow)?;
        if allocation_bytes::<u32>(backing_words)? > isize::MAX as usize
            || allocation_bytes::<Element>(max_entries)? > isize::MAX as usize
        {
            return Err(HashConsCapacityError::CapacityOverflow);
        }
        let mut inner = VecHashCons::new();
        inner
            .backing
            .try_reserve_exact(backing_words)
            .map_err(HashConsCapacityError::VectorAllocation)?;
        inner
            .elements
            .try_reserve_exact(max_entries)
            .map_err(HashConsCapacityError::VectorAllocation)?;
        inner
            .table
            .try_reserve(max_entries, |_| 0)
            .map_err(HashConsCapacityError::TableAllocation)?;
        // Initialize the full writable backing while still in preparation.
        // Insertion only copies into this slice; it never resizes or reserves.
        inner.backing.resize(backing_words, 0);
        inner.retained_capacity_bytes()?;
        Ok(Self {
            inner,
            max_words,
            max_entries,
            max_encoded_words,
            insertion_active: false,
            backing_funding: None,
        })
    }

    /// Starts an encoding into the uncommitted backing tail. An optional paid
    /// backing owner reserves reached scratch growth before any word is written.
    /// All staging geometry is checked before any word is written. The writer
    /// must produce exactly `encoded_words`; unique-entry limits are checked
    /// after duplicate lookup, before publication.
    ///
    /// Drop cancels and clears written scratch. Deliberately forgetting the
    /// guard leaves insertion fenced until this owner is dropped; read-only
    /// lookup remains available. There is no unchecked recovery or reserve API.
    pub fn begin_insert(
        &mut self,
        encoded_words: usize,
    ) -> Result<PreparedInsertion<'_>, HashConsCapacityError> {
        if self.insertion_active {
            return Err(HashConsCapacityError::InsertionInProgress);
        }
        if self.inner.curr_elt.backing_end != 0 {
            return Err(HashConsCapacityError::InvalidPreparedStorage);
        }
        let start = self.inner.curr_elt.backing_start as usize;
        let committed_words = start
            .checked_sub(4)
            .ok_or(HashConsCapacityError::InvalidPreparedStorage)?;
        let end = start
            .checked_add(encoded_words)
            .ok_or(HashConsCapacityError::CapacityOverflow)?;
        u32::try_from(end).map_err(|_| HashConsCapacityError::IndexOverflow)?;
        u32::try_from(self.inner.elements.len())
            .map_err(|_| HashConsCapacityError::IndexOverflow)?;
        if encoded_words > self.max_encoded_words && self.backing_funding.is_some() {
            self.grow_backing(self.max_words, encoded_words)?;
        }
        if encoded_words > self.max_encoded_words {
            return Err(HashConsCapacityError::ScratchExceeded {
                required_words: encoded_words,
                capacity_words: self.max_encoded_words,
            });
        }
        if committed_words > self.max_words
            || end > self.inner.backing.len()
            || self.inner.elements.len() > self.max_entries
            || self.inner.table.len() != self.inner.elements.len()
            || self.inner.elements.len() > self.inner.table.capacity()
        {
            return Err(HashConsCapacityError::InvalidPreparedStorage);
        }
        self.insertion_active = true;
        Ok(PreparedInsertion {
            owner: self,
            start,
            expected_words: encoded_words,
            written_words: 0,
            failure: None,
        })
    }

    /// Returns an existing id even when every prepared allowance is exhausted.
    /// For a new vector, capacity checks and any explicitly funded storage
    /// growth precede publication. Failed backing growth stays in this owner;
    /// an escaped funding error retains its separate account owner.
    pub fn try_insert(&mut self, data: &[u32]) -> Result<u32, HashConsCapacityError> {
        if self.insertion_active {
            return Err(HashConsCapacityError::InsertionInProgress);
        }
        if let Some(id) = self.inner.lookup(data) {
            return Ok(id);
        }
        if self.inner.curr_elt.backing_end != 0 {
            return Err(HashConsCapacityError::InvalidPreparedStorage);
        }
        let start = self.inner.curr_elt.backing_start as usize;
        let end = start
            .checked_add(data.len())
            .ok_or(HashConsCapacityError::CapacityOverflow)?;
        let (id, end_u32) = self.validate_new_entry(end)?;
        let hash = hash_slice(&self.inner.hasher, data);
        self.inner.backing[start..end].copy_from_slice(data);
        self.publish_entry(id, end_u32, hash);
        Ok(id)
    }

    fn validate_new_entry(&mut self, end: usize) -> Result<(u32, u32), HashConsCapacityError> {
        let end_u32 = u32::try_from(end).map_err(|_| HashConsCapacityError::IndexOverflow)?;
        let required_words = end
            .checked_sub(4)
            .ok_or(HashConsCapacityError::InvalidPreparedStorage)?;
        if required_words > self.max_words && self.backing_funding.is_some() {
            self.grow_backing(required_words, self.max_encoded_words)?;
        }
        if required_words > self.max_words {
            return Err(HashConsCapacityError::WordsExceeded {
                required_words,
                capacity_words: self.max_words,
            });
        }
        let required_entries = self
            .inner
            .elements
            .len()
            .checked_add(1)
            .ok_or(HashConsCapacityError::CapacityOverflow)?;
        let id = u32::try_from(self.inner.elements.len())
            .map_err(|_| HashConsCapacityError::IndexOverflow)?;
        u32::try_from(required_entries).map_err(|_| HashConsCapacityError::IndexOverflow)?;
        if required_entries > self.max_entries && self.backing_funding.is_some() {
            self.grow_entries(required_entries)?;
        }
        if required_entries > self.max_entries {
            return Err(HashConsCapacityError::EntriesExceeded {
                required_entries,
                capacity_entries: self.max_entries,
            });
        }
        if end > self.inner.backing.len()
            || required_entries > self.inner.elements.capacity()
            || self.inner.table.len() != self.inner.elements.len()
            || required_entries > self.inner.table.capacity()
        {
            return Err(HashConsCapacityError::InvalidPreparedStorage);
        }
        Ok((id, end_u32))
    }

    // All callers validate geometry/capacity and find duplicates before this
    // infallible publication. No caller-supplied callback or allocator runs.
    fn publish_entry(&mut self, id: u32, end_u32: u32, hash: u64) {
        self.inner.elements.push(Element {
            backing_start: self.inner.curr_elt.backing_start,
            backing_end: end_u32,
        });
        // This table never removes entries. capacity() guarantees insertion
        // without reallocation; the hasher closure is only a rehash fallback.
        // All data, indices and table capacity were validated before this point.
        let VecHashCons {
            table,
            backing,
            elements,
            hasher,
            curr_elt,
        } = &mut self.inner;
        table.insert_unique(hash, id, |id| {
            hash_slice(hasher, &backing[elements[*id as usize].as_range()])
        });
        curr_elt.backing_start = end_u32;
    }

    /// Finds an existing vector without changing prepared storage.
    pub fn lookup(&self, data: &[u32]) -> Option<u32> {
        self.inner.lookup(data)
    }
    /// Borrows an interned vector; panics when its id is not valid.
    pub fn get(&self, id: u32) -> &[u32] {
        self.inner.get(id)
    }
    /// Whether this owner contains the given id.
    pub fn is_valid(&self, id: u32) -> bool {
        self.inner.is_valid(id)
    }
    /// Number of unique vectors already inserted.
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    /// Whether no vectors have been inserted.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    /// Prepared limit on unique payload words, excluding the four-word prefix.
    pub fn max_words(&self) -> usize {
        self.max_words
    }
    /// Prepared limit on unique vectors, including an empty vector when present.
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }
    /// Maximum words in one incremental encoding, including duplicates.
    pub fn max_encoded_words(&self) -> usize {
        self.max_encoded_words
    }
    /// Exact current vector/table capacity. A separately paid backing callback
    /// owner is accounted by its own constructor, not included in this storage sum.
    pub fn retained_capacity_bytes(&self) -> Result<usize, HashConsCapacityError> {
        self.inner.retained_capacity_bytes()
    }
}

impl fmt::Debug for PreparedVecHashCons {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedVecHashCons")
            .field("entries", &self.len())
            .field("max_words", &self.max_words)
            .field("max_entries", &self.max_entries)
            .field("max_encoded_words", &self.max_encoded_words)
            .field("insertion_active", &self.insertion_active)
            .finish()
    }
}

#[derive(Clone, Copy, Debug)]
enum InsertionFailure {
    CapacityOverflow,
    EncodedLengthMismatch {
        expected_words: usize,
        actual_words: usize,
    },
}

impl InsertionFailure {
    fn error(self) -> HashConsCapacityError {
        match self {
            Self::CapacityOverflow => HashConsCapacityError::CapacityOverflow,
            Self::EncodedLengthMismatch {
                expected_words,
                actual_words,
            } => HashConsCapacityError::EncodedLengthMismatch {
                expected_words,
                actual_words,
            },
        }
    }
}

/// Exclusive writer for one prepared hash-consing insertion. Pushes never
/// allocate; finish can use an explicitly bound paid backing-growth owner.
///
/// Words go directly into reserved backing and become the committed value on
/// successful unique insertion. Duplicate, incomplete, rejected, abandoned and
/// unwound writers clear their written scratch. A failed push is sticky: later
/// pushes and `finish` return the first failure instead of publishing a prefix.
#[must_use = "finish the insertion or drop it to cancel and clear its scratch"]
pub struct PreparedInsertion<'a> {
    owner: &'a mut PreparedVecHashCons,
    start: usize,
    expected_words: usize,
    written_words: usize,
    failure: Option<InsertionFailure>,
}

impl PreparedInsertion<'_> {
    /// Writes one word after checking the declared encoding length.
    pub fn push_u32(&mut self, word: u32) -> Result<(), HashConsCapacityError> {
        self.push_slice(std::slice::from_ref(&word))
    }

    /// Writes borrowed words without allocating or copying existing entries.
    /// An overlong write changes no published data and clears prior scratch.
    pub fn push_slice(&mut self, words: &[u32]) -> Result<(), HashConsCapacityError> {
        if let Some(failure) = self.failure {
            return Err(failure.error());
        }
        let Some(next_words) = self.written_words.checked_add(words.len()) else {
            return Err(self.fail(InsertionFailure::CapacityOverflow));
        };
        if next_words > self.expected_words {
            return Err(self.fail(InsertionFailure::EncodedLengthMismatch {
                expected_words: self.expected_words,
                actual_words: next_words,
            }));
        }
        // begin_insert validated the entire declared region and holds exclusive
        // custody. This checked prefix cannot exceed that physical region.
        let from = self.start + self.written_words;
        let to = self.start + next_words;
        self.owner.inner.backing[from..to].copy_from_slice(words);
        self.written_words = next_words;
        Ok(())
    }

    /// Looks up the complete encoding before unique-capacity checks, then
    /// publishes these same staged words without a second payload copy.
    pub fn finish(mut self) -> Result<u32, HashConsCapacityError> {
        if let Some(failure) = self.failure {
            return Err(failure.error());
        }
        if self.written_words != self.expected_words {
            return Err(self.fail(InsertionFailure::EncodedLengthMismatch {
                expected_words: self.expected_words,
                actual_words: self.written_words,
            }));
        }
        let end = self.start + self.written_words;
        let staged = &self.owner.inner.backing[self.start..end];
        if let Some(id) = self.owner.inner.lookup(staged) {
            return Ok(id);
        }
        let hash = hash_slice(&self.owner.inner.hasher, staged);
        let (id, end_u32) = self.owner.validate_new_entry(end)?;
        self.owner.publish_entry(id, end_u32, hash);
        // These words are now committed and must survive guard cleanup.
        self.written_words = 0;
        Ok(id)
    }

    fn fail(&mut self, failure: InsertionFailure) -> HashConsCapacityError {
        self.clear_scratch();
        self.failure = Some(failure);
        failure.error()
    }

    fn clear_scratch(&mut self) {
        self.owner.inner.backing[self.start..self.start + self.written_words].fill(0);
        self.written_words = 0;
    }
}

impl Drop for PreparedInsertion<'_> {
    fn drop(&mut self) {
        self.clear_scratch();
        self.owner.insertion_active = false;
    }
}

impl fmt::Debug for PreparedInsertion<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedInsertion")
            .field("expected_words", &self.expected_words)
            .field("written_words", &self.written_words)
            .field("failure", &self.failure)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod incremental_tests;

pub use source_copy::{
    HashConsCopyFailure, HashConsCopyPlan, HashConsCopyRequirements, HashConsEmptySourcePlan,
    HashConsPreparedSourcePlan, HashConsPreparedSourceRequirements,
};

#[cfg(test)]
#[path = "hashcons/entry_growth_tests.rs"]
mod entry_growth_tests;

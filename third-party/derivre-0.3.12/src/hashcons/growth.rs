//! Closed paid growth for reached hash-cons backing and entry allocations.
use super::{HashConsCapacityError, PreparedVecHashCons, VecHashCons};
use crate::{ParserAllocationFailure, ParserAllocationFunding};
use std::{alloc::Layout, mem::{size_of, size_of_val}};

impl VecHashCons {
    // Shared initialization worker. The ordinary caller retains its existing
    // growth policy; the paid caller supplies exact reached physical geometry.
    pub(super) fn initialize_backing_with<E>(
        &mut self,
        total: usize,
        reserve: impl FnOnce(&mut Vec<u32>, usize) -> Result<(), E>,
    ) -> Result<(), E> {
        if self.backing.len() < total {
            reserve(&mut self.backing, total)?;
            self.backing.resize(total, 0);
        }
        Ok(())
    }
}
impl PreparedVecHashCons {
    pub(crate) fn backing_funding(&self) -> Option<&ParserAllocationFunding> {
        self.backing_funding.as_ref()
    }

    /// Starts the shared guarded insertion worker from its intrinsic empty
    /// representation. The four-word alignment prefix is funded before birth;
    /// entries and encoding scratch grow only when an insertion reaches them.
    pub(crate) fn empty_with_funding(
        funding: ParserAllocationFunding,
    ) -> Result<Self, HashConsCapacityError> {
        let mut value = Self {
            inner: VecHashCons::new(),
            max_words: 0,
            max_entries: 0,
            max_encoded_words: 0,
            insertion_active: false,
            backing_funding: Some(funding),
        };
        value.grow_backing(0, 0)?;
        Ok(value)
    }

    /// Moves an independently copied, quiescent source into the same mutable
    /// representation. Its existing backing is already owned by the copy plan;
    /// this operation allocates nothing and invents no future capacity.
    pub(crate) fn from_copied_source(
        inner: VecHashCons,
        source_words: usize,
        source_encoding: usize,
        funding: ParserAllocationFunding,
    ) -> Self {
        assert_eq!(inner.curr_elt.backing_end, 0);
        assert_eq!(inner.backing.len(), source_words + 4 + source_encoding);
        let max_words = source_words;
        let max_entries = inner.elements.len();
        Self {
            inner, max_words, max_entries, max_encoded_words: source_encoding,
            insertion_active: false, backing_funding: Some(funding),
        }
    }

    /// Explicit capacity requests use the same paid physical growth as reached
    /// insertions. This reserves actual storage, not a claim about future work.
    pub(crate) fn reserve_additional(&mut self, entries: usize) -> Result<(), HashConsCapacityError> {
        let requested_entries = self.inner.elements.len().checked_add(entries)
            .ok_or(HashConsCapacityError::CapacityOverflow)?;
        let words = entries.checked_mul(4).and_then(|n| self.max_words.checked_add(n))
            .ok_or(HashConsCapacityError::CapacityOverflow)?;
        self.grow_backing(words, self.max_encoded_words)?;
        self.grow_entries(requested_entries.max(self.max_entries))
    }

    /// Attaches one already paid owner to this actual quiescent destination.
    /// The same owner funds reached backing and source-quoted table entry growth.
    /// Source copies do not inherit it, and no ordinary mutation is exposed.
    pub fn bind_backing_funding(
        &mut self,
        funding: ParserAllocationFunding,
    ) -> Result<(), HashConsCapacityError> {
        if self.insertion_active
            || self.backing_funding.is_some()
            || self.inner.curr_elt.backing_end != 0
        {
            return Err(HashConsCapacityError::InvalidPreparedStorage);
        }
        self.backing_funding = Some(funding);
        Ok(())
    }
    pub(super) fn grow_backing(
        &mut self,
        words: usize,
        scratch: usize,
    ) -> Result<(), HashConsCapacityError> {
        let funding = self
            .backing_funding
            .as_ref()
            .ok_or(HashConsCapacityError::InvalidPreparedStorage)?
            .clone();
        let words = words.max(self.max_words);
        let scratch = scratch.max(self.max_encoded_words);
        let total = words
            .checked_add(4)
            .and_then(|n| n.checked_add(scratch))
            .ok_or(HashConsCapacityError::CapacityOverflow)?;
        u32::try_from(total).map_err(|_| HashConsCapacityError::IndexOverflow)?;
        let layout =
            Layout::array::<u32>(total).map_err(|_| HashConsCapacityError::CapacityOverflow)?;
        let parts = [
            size_of::<Self>(),
            size_of::<ParserAllocationFunding>(),
            size_of::<ParserAllocationFailure>(),
            size_of::<HashConsCapacityError>(),
            size_of::<(&mut VecHashCons, &ParserAllocationFunding, usize)>(),
            size_of::<(&mut Vec<u32>, usize)>(),
            size_of::<(usize, usize, usize)>(),
            size_of::<Layout>(),
            size_of::<Result<(), HashConsCapacityError>>(),
            size_of::<Result<(), std::collections::TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<Result<(), ()>>(),
        ];
        funding.reserve(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(HashConsCapacityError::CapacityOverflow)?,
        )?;
        self.inner
            .initialize_backing_with(total, |backing, total| {
                if backing.capacity() < total {
                    funding.reserve(layout.size())?;
                    backing
                        .try_reserve_exact(total - backing.len())
                        .map_err(HashConsCapacityError::VectorAllocation)?;
                    if backing.capacity() != total {
                        return Err(HashConsCapacityError::InvalidPreparedStorage);
                    }
                }
                Ok(())
            })?;
        self.max_words = words;
        self.max_encoded_words = scratch;
        Ok(())
    }
}

impl PreparedVecHashCons {
    pub(super) fn grow_entries(&mut self, entries: usize) -> Result<(), HashConsCapacityError> {
        use super::{hash_slice, Element};
        let funding = self.backing_funding.as_ref()
            .ok_or(HashConsCapacityError::InvalidPreparedStorage)?.clone();
        u32::try_from(entries).map_err(|_| HashConsCapacityError::IndexOverflow)?;
        let additional = entries.checked_sub(self.inner.elements.len())
            .ok_or(HashConsCapacityError::InvalidPreparedStorage)?;
        if self.inner.table.len() != self.inner.elements.len() {
            return Err(HashConsCapacityError::InvalidPreparedStorage);
        }
        // This exact query shares hashbrown's real resize/in-place-rehash choice
        // and TableLayout. It performs no allocation, hashing, or mutation.
        let table_layout = self.inner.table.try_reserve_layout(additional)
            .map_err(HashConsCapacityError::TableAllocation)?;
        let element_layout = if entries > self.inner.elements.capacity() {
            Some(Layout::array::<Element>(entries).map_err(|_| HashConsCapacityError::CapacityOverflow)?)
        } else { None };
        let frames = [
            size_of::<Self>(), size_of::<ParserAllocationFunding>(),
            size_of::<ParserAllocationFailure>(), size_of::<HashConsCapacityError>(),
            size_of::<(&mut Self, usize)>(), size_of::<hashbrown::HashTable<u32>>(),
            size_of::<Vec<Element>>(), size_of::<Layout>() * 2,
            size_of::<Option<Layout>>() * 2, size_of::<(usize, usize)>(),
            size_of::<Result<(), HashConsCapacityError>>(),
            size_of::<Result<(), std::collections::TryReserveError>>(),
            size_of::<Result<(), hashbrown::TryReserveError>>(),
            size_of::<Result<Option<Layout>, hashbrown::TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<Option<usize>>(), size_of::<Result<u32, std::num::TryFromIntError>>(),
        ];
        let bytes = frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
            .and_then(|n| n.checked_add(element_layout.map_or(0, |layout| layout.size())))
            .and_then(|n| n.checked_add(table_layout.map_or(0, |layout| layout.size())))
            .ok_or(HashConsCapacityError::CapacityOverflow)?;
        // New requested allocations are fully prepaid while prior allocations
        // remain alive. A later allocation failure keeps accepted capacity in
        // this owner and does not publish the logical entry allowance or an ID.
        funding.reserve(bytes)?;
        if element_layout.is_some() {
            self.inner.elements.try_reserve_exact(additional)
                .map_err(HashConsCapacityError::VectorAllocation)?;
            if self.inner.elements.capacity() != entries {
                return Err(HashConsCapacityError::InvalidPreparedStorage);
            }
        }
        let VecHashCons { table, elements, backing, hasher, .. } = &mut self.inner;
        let rehash = |id: &u32| hash_slice(hasher, &backing[elements[*id as usize].as_range()]);
        funding.reserve(size_of_val(&rehash).checked_add(size_of::<(&mut hashbrown::HashTable<u32>, usize)>())
            .ok_or(HashConsCapacityError::CapacityOverflow)?)?;
        table.try_reserve(additional, rehash).map_err(HashConsCapacityError::TableAllocation)?;
        // The ordinary global allocator's logical allocation size equals its
        // request; this verifies the source before the new entry is published.
        if table_layout.is_some_and(|layout| table.allocation_size() != layout.size())
            || entries > table.capacity() {
            return Err(HashConsCapacityError::InvalidPreparedStorage);
        }
        self.max_entries = entries;
        Ok(())
    }
}

impl PreparedVecHashCons {
    // Same bound source owner as actual arena growth. A detached fixed copy has
    // no authority to grow a derivative, traversal or weight destination.
    pub(crate) fn grow_workspace<T>(
        &self,
        values: &mut Vec<T>,
        total: usize,
    ) -> Result<(), HashConsCapacityError> {
        if total <= values.capacity() {
            return Ok(());
        }
        let funding = self
            .backing_funding
            .as_ref()
            .ok_or(HashConsCapacityError::InvalidPreparedStorage)?;
        let layout =
            Layout::array::<T>(total).map_err(|_| HashConsCapacityError::CapacityOverflow)?;
        let frames = [
            size_of::<(&Self, &mut Vec<T>, usize)>(),
            size_of::<Layout>(),
            size_of::<Option<usize>>(),
            size_of::<HashConsCapacityError>(),
            size_of::<Result<(), HashConsCapacityError>>(),
            size_of::<Result<(), std::collections::TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
        ];
        let bytes = frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .and_then(|n| n.checked_add(layout.size()))
            .ok_or(HashConsCapacityError::CapacityOverflow)?;
        // Charge the complete new request while the previous backing is alive.
        // Failed reallocation preserves the actual original vector and account.
        funding.reserve(bytes)?;
        values
            .try_reserve_exact(total - values.len())
            .map_err(HashConsCapacityError::VectorAllocation)?;
        if values.capacity() != total {
            return Err(HashConsCapacityError::InvalidPreparedStorage);
        }
        Ok(())
    }
}

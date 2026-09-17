//! Closed paid owner for reached hash-cons backing and entry allocations.
use super::{HashConsCapacityError, PreparedVecHashCons, VecHashCons};
use std::{
    alloc::Layout,
    error::Error,
    fmt,
    mem::{size_of, size_of_val},
    sync::{atomic::AtomicUsize, Arc, OnceLock},
};
trait Account: fmt::Debug + Send + Sync {
    fn reserve(&self, bytes: usize) -> bool;
    fn cause(&self) -> Option<&(dyn Error + 'static)>;
    fn retire(self: Arc<Self>);
}
struct Payload<F, E> {
    failure: OnceLock<E>,
    funding: F,
}
impl<F, E> fmt::Debug for Payload<F, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HashConsFunding")
            .field("failed", &self.failure.get().is_some())
            .finish()
    }
}
impl<F, E> Account for Payload<F, E>
where
    F: Fn(usize) -> Result<(), E> + Send + Sync + 'static,
    E: Error + Send + Sync + 'static,
{
    fn reserve(&self, bytes: usize) -> bool {
        if self.failure.get().is_some() {
            return false;
        }
        match (self.funding)(bytes) {
            Ok(()) => true,
            Err(error) => {
                let _ = self.failure.set(error);
                false
            }
        }
    }
    fn cause(&self) -> Option<&(dyn Error + 'static)> {
        self.failure.get().map(|e| e as _)
    }
    fn retire(self: Arc<Self>) {
        if let Some(payload) = Arc::into_inner(self) {
            drop(payload);
        }
    }
}
/// Independently paid callback/first-failure storage. It supplies reached storage
/// bytes only; the enclosing actual prepared table supplies source and insertion order.
pub struct PreparedHashConsFunding(Option<Arc<dyn Account>>);
impl Clone for PreparedHashConsFunding {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Drop for PreparedHashConsFunding {
    fn drop(&mut self) {
        if let Some(account) = self.0.take() {
            account.retire();
        }
    }
}
impl fmt::Debug for PreparedHashConsFunding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedHashConsFunding")
            .field(
                "failed",
                &self.0.as_ref().is_some_and(|a| a.cause().is_some()),
            )
            .finish()
    }
}
/// Inline construction refusal; no callback destination exists on failure.
#[derive(Debug)]
pub enum HashConsFundingPreparationError<E> {
    /// The actual shared callback/error layout cannot be represented.
    Overflow,
    /// The real account refused before allocation.
    Funding(E),
}
impl<E: fmt::Display> fmt::Display for HashConsFundingPreparationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => f.write_str("hash-cons funding owner layout overflow"),
            Self::Funding(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl<E: Error + 'static> Error for HashConsFundingPreparationError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Funding(e) => Some(e),
            _ => None,
        }
    }
}
/// The original concrete refusal remains borrowed from its fixed first-error
/// slot. The final alias frees its Arc shell before dropping error/source/H.
#[derive(Debug)]
pub struct HashConsFundingFailure {
    funding: PreparedHashConsFunding,
}
impl fmt::Display for HashConsFundingFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.source() {
            Some(cause) => fmt::Display::fmt(cause, f),
            None => f.write_str("hash-cons funding is unavailable"),
        }
    }
}
impl Error for HashConsFundingFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.funding.0.as_ref().and_then(|a| a.cause())
    }
}
impl PreparedHashConsFunding {
    /// Exact prospective constructor payment for this actual retained callback
    /// and its error slot. Inspection invokes no callback or allocation.
    pub fn preparation_bytes<F, E>(_: &F) -> Option<usize>
    where F: Fn(usize) -> Result<(), E> + Send + Sync + 'static,
        E: Error + Send + Sync + 'static,
    {
        let layout = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Payload<F, E>>())
            .ok()?
            .0
            .pad_to_align();
        let parts = [
            layout.size(),
            size_of::<Payload<F, E>>(),
            size_of::<F>(),
            size_of::<OnceLock<E>>(),
            size_of::<Self>(),
            size_of::<Arc<Payload<F, E>>>(),
            size_of::<Arc<dyn Account>>(),
            size_of::<HashConsFundingPreparationError<E>>(),
            size_of::<HashConsFundingFailure>(),
            size_of::<Result<Self, HashConsFundingPreparationError<E>>>(),
            size_of::<Result<(), E>>(),
            size_of::<Layout>(),
            size_of::<Option<usize>>(),
            size_of::<Result<E, E>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Reserves the actual callback/error slot and shared shell before allocation.
    /// The callback must retain its real cumulative funding account.
    pub fn prepare<F, E>(funding: F) -> Result<Self, HashConsFundingPreparationError<E>>
    where
        F: Fn(usize) -> Result<(), E> + Send + Sync + 'static,
        E: Error + Send + Sync + 'static,
    {
        let bytes = Self::preparation_bytes(&funding)
            .ok_or(HashConsFundingPreparationError::Overflow)?;
        funding(bytes).map_err(HashConsFundingPreparationError::Funding)?;
        Ok(Self(Some(Arc::new(Payload {
            failure: OnceLock::new(),
            funding,
        }))))
    }
    fn reserve(&self, bytes: usize) -> Result<(), HashConsCapacityError> {
        if self.0.as_ref().is_some_and(|a| a.reserve(bytes)) {
            Ok(())
        } else {
            Err(HashConsCapacityError::Funding(HashConsFundingFailure {
                funding: self.clone(),
            }))
        }
    }
}
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
    /// Attaches one already paid owner to this actual quiescent destination.
    /// The same owner funds reached backing and source-quoted table entry growth.
    /// Source copies do not inherit it, and no ordinary mutation is exposed.
    pub fn bind_backing_funding(
        &mut self,
        funding: PreparedHashConsFunding,
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
            size_of::<PreparedHashConsFunding>(),
            size_of::<HashConsFundingFailure>(),
            size_of::<HashConsCapacityError>(),
            size_of::<(&mut VecHashCons, &PreparedHashConsFunding, usize)>(),
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
            size_of::<Self>(), size_of::<PreparedHashConsFunding>(),
            size_of::<HashConsFundingFailure>(), size_of::<HashConsCapacityError>(),
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

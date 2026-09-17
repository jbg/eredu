//! Closed fixed-slot initialization, without copying arbitrary element payloads.

use super::{HostSlotMetadata, HostSlotTable};
use std::{fmt, iter::FusedIterator, marker::PhantomData, mem::size_of};

/// The actual source or requested fixed destination has no representable extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum HostSlotInitializationError {
    /// The existing source table has no complete checked inline capacity.
    #[error("host slot source has an unknown inline extent")]
    UnknownSourceExtent,
    /// A payload or explicit temporary sum cannot be represented.
    #[error("host slot {component} overflow")]
    Overflow {
        /// The quantity that overflowed or exceeded the allocation limit.
        component: &'static str,
    },
}

/// Metadata-only preparation coupled to an actual immutable table borrow.
///
/// Borrows actual source values S and plans one fixed Option<D> destination,
/// including vacant cells. D defaults to S for same-type copying. It never
/// calls Clone, Default, or an initializer callback. The allocating worker
/// is crate-private: this public plan grants no construction, funding, nested
/// resource copying, or execution authority.
///
/// Measures requested inline slot payload and explicit value temporaries only.
/// Nested resources, allocator overhead, ownership/registry bookkeeping, and
/// general compiler stack frames require separate contracts. S and D can have
/// interior mutability; borrowed access does not promise transitive immutability.
#[must_use = "this diagnostic plan grants no allocation or copy authority"]
pub struct HostSlotInitialization<'a, S, D = S> {
    source: SourceSlots<'a, S>,
    extent: InitializationExtent,
    // There is no destination value or ownership until the private worker runs.
    destination: PhantomData<fn() -> D>,
}

enum SourceSlots<'a, T> {
    Original(&'a HostSlotTable<T>),
    Initialized(&'a InitializedHostSlots<T>),
}
impl<T> Copy for SourceSlots<'_, T> {}
impl<T> Clone for SourceSlots<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

#[derive(Clone, Copy, Debug)]
struct InitializationExtent {
    len: usize,
    retained: u64,
    peak: u64,
}
impl InitializationExtent {
    fn for_type<T>(len: usize) -> Result<Self, HostSlotInitializationError> {
        let slot_size = size_of::<Option<T>>();
        let retained = len
            .checked_mul(slot_size)
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(HostSlotInitializationError::Overflow {
                component: "destination payload",
            })?;
        // One fixed buffer transfers into Box. Two slot-sized moved values
        // cover the None argument/move and later T plus its enclosing Some(T).
        // This is explicit algorithm scratch, not compiler-frame accounting.
        let temporary = if len == 0 {
            0
        } else {
            u64::try_from(slot_size)
                .ok()
                .and_then(|bytes| bytes.checked_mul(2))
                .ok_or(HostSlotInitializationError::Overflow {
                    component: "slot value temporaries",
                })?
        };
        let peak =
            retained
                .checked_add(temporary)
                .ok_or(HostSlotInitializationError::Overflow {
                    component: "initialization peak",
                })?;
        Ok(Self {
            len,
            retained,
            peak,
        })
    }
}

impl<T> HostSlotTable<T> {
    /// Prepares a vacant destination from this actual table's extent.
    /// This borrows values without cloning them or granting allocation rights.
    pub fn prepare_copy_slots(
        &self,
    ) -> Result<HostSlotInitialization<'_, T>, HostSlotInitializationError> {
        HostSlotInitialization::prepare(SourceSlots::Original(self))
    }
}

impl<'a, S, D> HostSlotInitialization<'a, S, D> {
    fn prepare(source: SourceSlots<'a, S>) -> Result<Self, HostSlotInitializationError> {
        let (len, metadata) = match source {
            SourceSlots::Original(table) => (table.len(), table.metadata()),
            SourceSlots::Initialized(table) => (table.len(), table.metadata()),
        };
        metadata
            .capacity_bytes()
            .ok_or(HostSlotInitializationError::UnknownSourceExtent)?;
        Ok(Self {
            source,
            extent: InitializationExtent::for_type::<D>(len)?,
            destination: PhantomData,
        })
    }
    /// Selects a different fixed destination representation from the same actual
    /// source borrow. Recomputes its checked Option<NewD> payload and temporary
    /// envelope without constructing a value, changing source identity/count,
    /// or granting authority to convert source values or allocate nested data.
    pub fn for_destination<NewD>(
        self,
    ) -> Result<HostSlotInitialization<'a, S, NewD>, HostSlotInitializationError> {
        HostSlotInitialization::prepare(self.source)
    }

    /// Exact source slot count, including absent values when S is optional.
    pub fn len(&self) -> usize {
        self.extent.len
    }
    /// Whether this plan creates an empty slot container.
    pub fn is_empty(&self) -> bool {
        self.extent.len == 0
    }
    /// Metadata of the actual borrowed source, not a supplied token. Its extent
    /// can differ from the destination's Option<D> slot extent.
    pub fn source_metadata(&self) -> &'a HostSlotMetadata {
        match self.source {
            SourceSlots::Original(table) => table.metadata(),
            SourceSlots::Initialized(table) => table.metadata(),
        }
    }
    /// Borrows an original value without cloning or reconstruction.
    pub fn source_at(&self, index: usize) -> Option<&'a S> {
        match self.source {
            SourceSlots::Original(table) => table.slots().get(index),
            SourceSlots::Initialized(table) => table.get(index),
        }
    }
    /// Exact requested final fixed-box payload, excluding nested resources.
    pub fn retained_bytes(&self) -> u64 {
        self.extent.retained
    }
    /// Payload plus two explicit moved slot values for nonempty initialization
    /// and fill. No second destination table is constructed.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.extent.peak
    }

    /// A closed runtime admission path must supply prior host funding. This
    /// worker neither obtains nor certifies any funding scope itself.
    pub(crate) fn preparation_control_bytes() -> Option<usize> {
        [
            HostSlotMetadata::prepared_host_control_bytes()?,
            size_of::<Self>(),
            size_of::<HostSlotInitializationBuilder<D>>(),
            size_of::<Vec<Option<D>>>(),
            size_of::<Box<[Option<D>]>>(),
            size_of::<HostSlotTable<Option<D>>>(),
            size_of::<
                Result<HostSlotInitializationBuilder<D>, crate::working_memory::WorkingMemoryError>,
            >(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    pub(crate) fn initialize_prepared(
        self,
        identity: super::HostMetadataIdentity,
        authority: &eredu_core::HostPreparationAuthority,
    ) -> HostSlotInitializationBuilder<D> {
        self.initialize_with(Some((identity, authority)))
    }

    pub(crate) fn initialize(self) -> HostSlotInitializationBuilder<D> {
        self.initialize_with(None)
    }

    fn initialize_with(
        self,
        prepared: Option<(
            super::HostMetadataIdentity,
            &eredu_core::HostPreparationAuthority,
        )>,
    ) -> HostSlotInitializationBuilder<D> {
        // Rust 1.98.0 RawVec::with_capacity requests Layout::array(n) and stores
        // cap=n. These n pushes cannot grow. Non-ZST len==cap skips shrinking
        // during boxing and transfers the buffer. ZST/empty payload allocates
        // nothing. This concrete path, not final Box length, proves the bound.
        let mut slots = Vec::<Option<D>>::with_capacity(self.extent.len);
        for _ in 0..self.extent.len {
            slots.push(None);
        }
        #[cfg(test)]
        let before = (slots.capacity(), slots.as_ptr() as usize);
        let slots = slots.into_boxed_slice();
        #[cfg(test)]
        tests::record_initialization(self.extent.len, before.0, before.1, slots.as_ptr() as usize);
        HostSlotInitializationBuilder {
            slots: match prepared {
                Some((identity, authority)) => {
                    HostSlotTable::new_prepared_host(slots, identity, authority)
                }
                None => HostSlotTable::new(slots),
            },
            initialized: 0,
        }
    }
}
impl<S, D> fmt::Debug for HostSlotInitialization<'_, S, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostSlotInitialization")
            .field("source", self.source_metadata())
            .field("len", &self.len())
            .field("retained_bytes", &self.retained_bytes())
            .field(
                "initialization_peak_bytes",
                &self.initialization_peak_bytes(),
            )
            .finish()
    }
}

/// Fixed vacant destination produced only by the crate-private initializer.
/// Filling moves values into existing storage without allocation and grants no
/// authority to construct nested resources. Partial values stay owned through
/// errors/unwind. No mutable slice, resize, Clone, or owning export exists.
#[must_use = "retain partial slots through the caller's copy/failure custody"]
pub struct HostSlotInitializationBuilder<T> {
    slots: HostSlotTable<Option<T>>,
    initialized: usize,
}
impl<T> HostSlotInitializationBuilder<T> {
    /// Total fixed count selected by the original initialization plan.
    pub fn len(&self) -> usize {
        self.slots.len()
    }
    /// Whether the destination has no slots.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
    /// Values already moved into consecutive slots.
    pub fn initialized_count(&self) -> usize {
        self.initialized
    }
    /// Actual metadata covers every cell, including vacant ones.
    pub fn metadata(&self) -> &HostSlotMetadata {
        self.slots.metadata()
    }
    /// Moves one value into the next vacant cell. A full destination returns
    /// the rejected value intact and leaves prior cells unchanged.
    pub fn push(&mut self, value: T) -> Result<(), HostSlotPushError<T>> {
        if self.initialized == self.slots.len() {
            return Err(HostSlotPushError {
                value,
                capacity: self.slots.len(),
            });
        }
        self.slots.slots_mut()[self.initialized] = Some(value);
        self.initialized += 1;
        Ok(())
    }
    /// Completes only after every slot has been filled. An incomplete error
    /// retains its builder for cleanup or completion, not a completed source.
    pub fn finish(self) -> Result<InitializedHostSlots<T>, HostSlotFinishError<T>> {
        if self.initialized != self.slots.len() {
            return Err(HostSlotFinishError { builder: self });
        }
        Ok(InitializedHostSlots { slots: self.slots })
    }
}
impl<T> fmt::Debug for HostSlotInitializationBuilder<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostSlotInitializationBuilder")
            .field("metadata", self.metadata())
            .field("initialized", &self.initialized)
            .finish()
    }
}

/// A full destination rejected one extra value without dropping it.
pub struct HostSlotPushError<T> {
    value: T,
    capacity: usize,
}
impl<T> HostSlotPushError<T> {
    /// Full destination capacity at rejection.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Returns the original value without copying or destroying it.
    pub fn into_value(self) -> T {
        self.value
    }
}
impl<T> fmt::Debug for HostSlotPushError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostSlotPushError")
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}
impl<T> fmt::Display for HostSlotPushError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "host slot destination is full at {} slots",
            self.capacity
        )
    }
}
impl<T> std::error::Error for HostSlotPushError<T> {}

/// An incomplete finish retains the actual partial destination.
pub struct HostSlotFinishError<T> {
    builder: HostSlotInitializationBuilder<T>,
}
impl<T> HostSlotFinishError<T> {
    /// Required count fixed by the original plan.
    pub fn expected(&self) -> usize {
        self.builder.len()
    }
    /// Count of already initialized values.
    pub fn initialized(&self) -> usize {
        self.builder.initialized_count()
    }
    /// Restores the same partial builder without allocation.
    pub fn into_builder(self) -> HostSlotInitializationBuilder<T> {
        self.builder
    }
}
impl<T> fmt::Debug for HostSlotFinishError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostSlotFinishError")
            .field("expected", &self.expected())
            .field("initialized", &self.initialized())
            .finish()
    }
}
impl<T> fmt::Display for HostSlotFinishError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "host slot destination has {} initialized values; expected {}",
            self.initialized(),
            self.expected()
        )
    }
}
impl<T> std::error::Error for HostSlotFinishError<T> {}

/// Completed fixed slot ownership with borrowed access only.
/// Retains the initializer's actual Option<T> buffer without conversion, cloning
/// or another allocation. Every outer option is filled; T may itself be optional
/// or internally mutable. No transitive copying/funding claim is made.
#[must_use = "retain slots for the full lifetime of their actual payload"]
pub struct InitializedHostSlots<T> {
    slots: HostSlotTable<Option<T>>,
}
impl<T> InitializedHostSlots<T> {
    /// Exact fixed number of values.
    pub fn len(&self) -> usize {
        self.slots.len()
    }
    /// Whether there are no values.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
    /// Borrows a value without cloning or exposing a mutable slot.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.slots.slots().get(index).map(|value| {
            value
                .as_ref()
                .expect("completed slots have no vacant cells")
        })
    }
    /// Visits all values, including inner None when T itself is optional.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> + DoubleEndedIterator + FusedIterator {
        self.slots.slots().iter().map(|value| {
            value
                .as_ref()
                .expect("completed slots have no vacant cells")
        })
    }
    /// Actual fixed destination extent and attachment custody.
    pub fn metadata(&self) -> &HostSlotMetadata {
        self.slots.metadata()
    }
    /// Plans another same-representation destination from this actual owner.
    /// Adds no nested Option layer and inherits no prior allocation authority.
    pub fn prepare_copy_slots(
        &self,
    ) -> Result<HostSlotInitialization<'_, T>, HostSlotInitializationError> {
        HostSlotInitialization::prepare(SourceSlots::Initialized(self))
    }
}
impl<T> fmt::Debug for InitializedHostSlots<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InitializedHostSlots")
            .field("metadata", self.metadata())
            .finish()
    }
}

#[cfg(test)]
mod tests;

mod dense;
pub use dense::{
    DenseHostSlotFinishError, DenseHostSlotInitialization, DenseHostSlotInitializationBuilder,
    InitializedDenseHostSlots, PreparedDenseHostCopyError,
};

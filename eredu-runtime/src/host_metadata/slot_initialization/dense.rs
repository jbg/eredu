//! One fixed dense destination buffer, prepared from an actual slot owner.

use super::{
    HostSlotInitialization, HostSlotInitializationError, HostSlotMetadata, HostSlotPushError,
    HostSlotTable, SourceSlots,
};
use std::{fmt, iter::FusedIterator, marker::PhantomData, mem::size_of};

mod prepared;
pub use prepared::PreparedDenseHostCopyError;

/// Source-borrowed preparation for a dense `[D]` destination.
///
/// The exact source owner supplies slot count and identity. Destination payload
/// is `len * size_of::<D>()`; the fill envelope also covers two explicit moved
/// D values for a nonempty table. This is not an arbitrary native constructor,
/// nested resource, allocator, bookkeeping or compiler-frame bound.
///
/// This diagnostic grants no allocation, funding, copying or execution rights.
/// Its allocating worker remains crate-private for a later closed admission.
/// Source and destination types may have interior mutability.
#[must_use = "this diagnostic plan grants no allocation or copy authority"]
pub struct DenseHostSlotInitialization<'a, S, D = S> {
    source: SourceSlots<'a, S>,
    extent: DenseExtent,
    destination: PhantomData<fn() -> D>,
}

#[derive(Clone, Copy, Debug)]
struct DenseExtent {
    len: usize,
    retained: u64,
    peak: u64,
}

impl DenseExtent {
    fn for_type<T>(len: usize) -> Result<Self, HostSlotInitializationError> {
        let slot = size_of::<T>();
        let retained = len
            .checked_mul(slot)
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(HostSlotInitializationError::Overflow {
                component: "dense destination payload",
            })?;
        // The caller-produced value and the explicit push/move are conservatively
        // counted alongside the fixed buffer. No Option<D> table is constructed.
        let temporary = if len == 0 {
            0
        } else {
            u64::try_from(slot)
                .ok()
                .and_then(|bytes| bytes.checked_mul(2))
                .ok_or(HostSlotInitializationError::Overflow {
                    component: "dense value temporaries",
                })?
        };
        let peak =
            retained
                .checked_add(temporary)
                .ok_or(HostSlotInitializationError::Overflow {
                    component: "dense initialization peak",
                })?;
        Ok(Self {
            len,
            retained,
            peak,
        })
    }
}

impl<'a, S, PreviousD> HostSlotInitialization<'a, S, PreviousD> {
    /// Selects dense D cells rather than vacant Option cells. Source values,
    /// count and metadata still come from this plan's exact original borrow;
    /// the previous destination type is neither allocated nor converted.
    pub fn for_dense_destination<D>(
        self,
    ) -> Result<DenseHostSlotInitialization<'a, S, D>, HostSlotInitializationError> {
        DenseHostSlotInitialization::prepare(self.source)
    }
}

impl<'a, S, D> DenseHostSlotInitialization<'a, S, D> {
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
            extent: DenseExtent::for_type::<D>(len)?,
            destination: PhantomData,
        })
    }

    /// Exact count selected by the actual source owner.
    pub fn len(&self) -> usize {
        self.extent.len
    }

    /// Whether no destination values are required.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Actual borrowed source metadata, which may have a different slot extent.
    pub fn source_metadata(&self) -> &'a HostSlotMetadata {
        match self.source {
            SourceSlots::Original(table) => table.metadata(),
            SourceSlots::Initialized(table) => table.metadata(),
        }
    }

    /// Borrows one original S without cloning or reconstructing it.
    pub fn source_at(&self, index: usize) -> Option<&'a S> {
        match self.source {
            SourceSlots::Original(table) => table.slots().get(index),
            SourceSlots::Initialized(table) => table.get(index),
        }
    }

    /// Requested final dense boxed payload; nested resources are separate.
    pub fn retained_bytes(&self) -> u64 {
        self.extent.retained
    }

    /// Fixed payload plus two moved D values for nonempty fill. The future
    /// admitted owner must retain this envelope throughout partial fill/errors.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.extent.peak
    }

    pub(crate) fn preparation_control_bytes() -> Option<usize> {
        [
            HostSlotMetadata::prepared_host_control_bytes()?,
            size_of::<Self>(),
            size_of::<DenseHostSlotInitializationBuilder<D>>(),
            size_of::<Vec<D>>(),
            size_of::<Box<[D]>>(),
            size_of::<HostSlotTable<D>>(),
            size_of::<Result<InitializedDenseHostSlots<D>, DenseHostSlotFinishError<D>>>(),
            size_of::<
                Option<(
                    super::super::HostMetadataIdentity,
                    &eredu_core::HostPreparationAuthority,
                )>,
            >(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    pub(crate) fn try_initialize_retaining_source(
        self,
    ) -> Result<(Self, DenseHostSlotInitializationBuilder<D>), std::collections::TryReserveError>
    {
        let mut slots = Vec::<D>::new();
        slots.try_reserve_exact(self.extent.len)?;
        #[cfg(test)]
        tests::record_initialization(self.extent.len, slots.capacity(), slots.as_ptr() as usize);
        let expected = self.extent.len;
        Ok((self, DenseHostSlotInitializationBuilder { slots, expected }))
    }

    /// Requires prior authority from a closed runtime admission path. This
    /// worker acquires no funding and cannot certify host or native completion.
    pub(crate) fn initialize(self) -> DenseHostSlotInitializationBuilder<D> {
        self.initialize_retaining_source().1
    }

    // Closed funded owners retain the actual source borrow through filling.
    // This uses the same one-buffer worker; it grants no repeatable public
    // allocation authority or numerical source-copy operation.
    pub(crate) fn initialize_retaining_source(
        self,
    ) -> (Self, DenseHostSlotInitializationBuilder<D>) {
        // Pinned Rust 1.98.0 RawVec requests exactly Layout::array::<D>(n) and
        // records n. Pushes stop at n, so no growth occurs. Exact finish has
        // len==cap for non-ZST values and into_boxed_slice transfers this buffer.
        // Empty and ZST payloads require no slot allocation. This derivation is
        // about the concrete worker, not an inference from final Box length.
        let slots = Vec::<D>::with_capacity(self.extent.len);
        #[cfg(test)]
        tests::record_initialization(self.extent.len, slots.capacity(), slots.as_ptr() as usize);
        let builder = DenseHostSlotInitializationBuilder {
            slots,
            expected: self.extent.len,
        };
        (self, builder)
    }
}

impl<S, D> fmt::Debug for DenseHostSlotInitialization<'_, S, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DenseHostSlotInitialization")
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

/// One fixed-capacity partial dense destination, constructed only privately.
///
/// Every pushed value remains owned on incomplete finish, failure and unwind.
/// No Clone, Default, initializer callback, resize, mutable slice or owning
/// export exists. There is no destination metadata token before exact finish;
/// later admitted custody must cover the full allocated capacity independently
/// of the initialized length and remain outside this payload's lifetime.
#[must_use = "retain partial slots through the caller's copy/failure custody"]
pub struct DenseHostSlotInitializationBuilder<T> {
    slots: Vec<T>,
    expected: usize,
}

impl<T> DenseHostSlotInitializationBuilder<T> {
    pub(crate) fn into_exact_values(self) -> Result<Vec<T>, Self> {
        if self.slots.len() == self.expected {
            Ok(self.slots)
        } else {
            Err(self)
        }
    }
    #[cfg(test)]
    pub(crate) fn capacity_overflow_for_test(&mut self) -> std::collections::TryReserveError {
        self.slots
            .try_reserve_exact(usize::MAX)
            .expect_err("actual capacity overflow")
    }
    pub(crate) fn into_partial_values(self) -> Vec<T> {
        self.slots
    }

    /// Required fixed count, independent of initialized values or ZST capacity.
    pub fn len(&self) -> usize {
        self.expected
    }

    /// Whether the fixed destination has no slots.
    pub fn is_empty(&self) -> bool {
        self.expected == 0
    }

    /// Number of consecutive initialized values already owned by this builder.
    pub fn initialized_count(&self) -> usize {
        self.slots.len()
    }

    /// Moves a value into the next slot without growing the fixed buffer.
    /// Full rejection preserves that value and every earlier initialized value.
    pub fn push(&mut self, value: T) -> Result<(), HostSlotPushError<T>> {
        if self.slots.len() == self.expected {
            return Err(HostSlotPushError {
                value,
                capacity: self.expected,
            });
        }
        self.slots.push(value);
        Ok(())
    }

    /// Finishes only at the exact count, transferring the same buffer to Box.
    /// Partial finish returns ownership of the original, unshrunk Vec in its
    /// error. It cannot publish a shorter completed table or copy its values.
    pub fn finish(self) -> Result<InitializedDenseHostSlots<T>, DenseHostSlotFinishError<T>> {
        self.finish_with_preparation(None)
    }
    pub(crate) fn finish_with_preparation(
        self,
        prepared: Option<(
            super::super::HostMetadataIdentity,
            &eredu_core::HostPreparationAuthority,
        )>,
    ) -> Result<InitializedDenseHostSlots<T>, DenseHostSlotFinishError<T>> {
        if self.slots.len() != self.expected {
            return Err(DenseHostSlotFinishError { builder: self });
        }
        let slots = self.slots.into_boxed_slice();
        #[cfg(test)]
        tests::record_boxed(slots.as_ptr() as usize);
        Ok(InitializedDenseHostSlots {
            slots: match prepared {
                Some((identity, host)) => HostSlotTable::new_prepared_host(slots, identity, host),
                None => HostSlotTable::new(slots),
            },
        })
    }
}

impl<T> fmt::Debug for DenseHostSlotInitializationBuilder<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DenseHostSlotInitializationBuilder")
            .field("expected", &self.expected)
            .field("initialized", &self.initialized_count())
            .finish()
    }
}

/// Incomplete dense finish retains the actual fixed-capacity partial buffer.
pub struct DenseHostSlotFinishError<T> {
    builder: DenseHostSlotInitializationBuilder<T>,
}

impl<T> DenseHostSlotFinishError<T> {
    /// Exact destination count selected by the source-bound plan.
    pub fn expected(&self) -> usize {
        self.builder.len()
    }

    /// Values retained in the partial buffer.
    pub fn initialized(&self) -> usize {
        self.builder.initialized_count()
    }

    /// Recovers the same partial builder without copying, shrinking or growing.
    pub fn into_builder(self) -> DenseHostSlotInitializationBuilder<T> {
        self.builder
    }
}

impl<T> fmt::Debug for DenseHostSlotFinishError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DenseHostSlotFinishError")
            .field("expected", &self.expected())
            .field("initialized", &self.initialized())
            .finish()
    }
}
impl<T> fmt::Display for DenseHostSlotFinishError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "dense host slot destination has {} initialized values; expected {}",
            self.initialized(),
            self.expected()
        )
    }
}
impl<T> std::error::Error for DenseHostSlotFinishError<T> {}

/// Completed fixed dense values with borrowed access only.
///
/// The exact initializer buffer is now an actual HostSlotTable<T>. No optional
/// slot wrapper, numerical clone, or second payload allocation is introduced.
/// T may have interior mutability; this wrapper does not assert otherwise.
#[must_use = "retain slots for the full lifetime of their actual payload"]
pub struct InitializedDenseHostSlots<T> {
    slots: HostSlotTable<T>,
}

impl<T> InitializedDenseHostSlots<T> {
    /// Fixed number of completed values.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether this destination has no values.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Borrows one value without a mutable or owning export.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.slots.slots().get(index)
    }

    /// Visits each actual dense value, including None when T itself is optional.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> + DoubleEndedIterator + FusedIterator {
        self.slots.slots().iter()
    }

    /// Actual dense table metadata, independently retained from source custody.
    pub fn metadata(&self) -> &HostSlotMetadata {
        self.slots.metadata()
    }

    /// Starts another diagnostic from the actual internal table. Select
    /// for_dense_destination again to preserve dense destination representation.
    /// Source values are T, not an added Option or wrapper type.
    pub fn prepare_copy_slots(
        &self,
    ) -> Result<HostSlotInitialization<'_, T>, HostSlotInitializationError> {
        self.slots.prepare_copy_slots()
    }

    /// Only a future closed runtime handoff may move this table alongside its
    /// already established host custody. This method grants no new authority.
    pub(crate) fn into_table(self) -> HostSlotTable<T> {
        self.slots
    }
}

impl<T> fmt::Debug for InitializedDenseHostSlots<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InitializedDenseHostSlots")
            .field("metadata", self.metadata())
            .finish()
    }
}

#[cfg(test)]
mod tests;

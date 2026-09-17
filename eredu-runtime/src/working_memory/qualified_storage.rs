//! Managed extents of the pinned fresh Global storage producer.
//!
//! The build qualification names the actual compiler/liballoc/libstd. It is not
//! a promise about malloc-private bookkeeping or process residency. Exposed Vec
//! capacity and Arc/Rc headers are managed extents and are included here.
use super::WorkingMemoryError;
use std::{alloc::Layout, collections::TryReserveError, mem::size_of};

include!(concat!(env!("OUT_DIR"), "/qualified_fresh_vec.rs"));

pub(super) fn qualified() -> bool {
    QUALIFIED_FRESH_VEC
}

pub(super) enum ReserveError {
    Unqualified,
    Layout,
    Reserve(TryReserveError),
    Contract,
}

/// Caller owns the empty destination across failure and already holds its
/// accepted charge. Exact reserve is qualified BEFORE allocation; the capacity
/// comparison is a defensive contradiction check, not a post-hoc bound.
pub(super) fn reserve_empty<T>(
    destination: &mut Vec<T>,
    elements: usize,
) -> Result<(), ReserveError> {
    if !qualified() {
        return Err(ReserveError::Unqualified);
    }
    if !destination.is_empty() || (size_of::<T>() != 0 && destination.capacity() != 0) {
        return Err(ReserveError::Contract);
    }
    Layout::array::<T>(elements).map_err(|_| ReserveError::Layout)?;
    destination
        .try_reserve_exact(elements)
        .map_err(ReserveError::Reserve)?;
    if size_of::<T>() != 0 && destination.capacity() != elements {
        return Err(ReserveError::Contract);
    }
    Ok(())
}

/// Only reached from a recipe whose complete finite managed contribution was
/// accepted. The legacy scalar-provider path preserves its existing producer.
pub(super) fn vector<T>(elements: usize, exact: bool) -> Result<Vec<T>, WorkingMemoryError> {
    if !exact {
        return Ok(Vec::with_capacity(elements));
    }
    let mut values = Vec::new();
    reserve_empty(&mut values, elements).map_err(|cause| match cause {
        ReserveError::Unqualified => WorkingMemoryError::UnknownBound,
        ReserveError::Layout => WorkingMemoryError::Overflow,
        ReserveError::Reserve(cause) => WorkingMemoryError::ControlStorageReserve(cause),
        ReserveError::Contract => WorkingMemoryError::IdentityMismatch,
    })?;
    Ok(values)
}

pub(super) fn array_bytes<T>(elements: usize) -> Result<u64, WorkingMemoryError> {
    Layout::array::<T>(elements)
        .ok()
        .and_then(|layout| u64::try_from(layout.size()).ok())
        .ok_or(WorkingMemoryError::Overflow)
}

/// Qualified ArcInner/RcInner are repr(C, align(2)) with two usize counters.
/// Both layouts use the same extent; atomics and Cells have usize layout on the
/// qualified target. The payload is included once, including final padding.
pub(crate) fn shared_bytes<T>() -> Result<u64, WorkingMemoryError> {
    shared_layout_bytes(Layout::new::<T>())
}

pub(crate) fn shared_layout_bytes(payload: Layout) -> Result<u64, WorkingMemoryError> {
    if !qualified() {
        return Err(WorkingMemoryError::UnknownBound);
    }
    Layout::new::<[usize; 2]>()
        .align_to(2)
        .map_err(|_| WorkingMemoryError::Overflow)?
        .pad_to_align()
        .extend(payload)
        .ok()
        .and_then(|(layout, _)| u64::try_from(layout.pad_to_align().size()).ok())
        .ok_or(WorkingMemoryError::Overflow)
}

pub(super) fn shared_header_bytes<T>() -> Result<u64, WorkingMemoryError> {
    shared_bytes::<T>()?
        .checked_sub(size_of::<T>() as u64)
        .ok_or(WorkingMemoryError::Overflow)
}

/// Named reserve/helper frames retained conservatively beside their destination.
/// The destination's heap is separately priced by its checked array layout.
pub(super) fn reserve_control_bytes<T>() -> Result<usize, WorkingMemoryError> {
    [
        size_of::<&mut Vec<T>>(),
        size_of::<usize>(),
        size_of::<Result<Layout, std::alloc::LayoutError>>(),
        size_of::<Result<(), ReserveError>>(),
        size_of::<ReserveError>(),
        size_of::<Result<(), TryReserveError>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .ok_or(WorkingMemoryError::Overflow)
}
pub(super) fn vector_control_bytes<T>() -> Result<u64, WorkingMemoryError> {
    reserve_control_bytes::<T>()?
        .checked_add(size_of::<Vec<T>>())
        .and_then(|n| n.checked_add(size_of::<Result<Vec<T>, WorkingMemoryError>>()))
        .and_then(|n| n.checked_add(size_of::<WorkingMemoryError>()))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(WorkingMemoryError::Overflow)
}

//! Shared paid copies of the actual initialized storage and retained capacity.
use std::{alloc::Layout, collections::TryReserveError, fmt, mem::{size_of, size_of_val}};
#[derive(Debug)]
pub(crate) enum Error<E> { Source, Overflow, Capacity, Funding(E), Allocation(TryReserveError) }
impl<E: fmt::Display> fmt::Display for Error<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source => f.write_str("mutable copy source differs"),
            Self::Overflow => f.write_str("mutable copy geometry overflow"),
            Self::Capacity => f.write_str("mutable copy destination differs"),
            Self::Funding(e) => fmt::Display::fmt(e, f),
            Self::Allocation(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for Error<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self { Self::Funding(e) => Some(e), Self::Allocation(e) => Some(e), _ => None }
    }
}
pub(crate) fn frame_bytes<T, E>() -> Option<usize> {
    // F is always Sized and borrowed. The callback's concrete representation
    // therefore does not change any transport paid by this shared worker.
    let parts = [size_of::<T>(), size_of::<&()>(), size_of::<Error<E>>(),
        size_of::<Result<(), Error<E>>>(), size_of::<Result<(), E>>()];
    parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
}
pub(crate) fn frame<T, F: Fn(usize) -> Result<(), E>, E>(funding: &F) -> Result<(), Error<E>> {
    funding(frame_bytes::<T, E>().ok_or(Error::Overflow)?).map_err(Error::Funding)
}
fn reserve_frame<T, E>() -> Option<usize> {
    frame_bytes::<(&mut Vec<T>, usize, Layout, Result<Layout, std::alloc::LayoutError>, Result<(), TryReserveError>), E>()
}
fn fixed_frame<T, E>() -> Option<usize> {
    frame_bytes::<(&mut Vec<T>, &Vec<T>, T, std::slice::Iter<'_, T>), E>()
}
fn vectors_frame<T, E>() -> Option<usize> {
    frame_bytes::<(&mut Vec<Vec<T>>, &Vec<Vec<T>>, std::slice::Iter<'_, Vec<T>>), E>()
}
fn optional_vectors_frame<T, E>() -> Option<usize> {
    frame_bytes::<(&mut Vec<Option<Vec<T>>>, &Vec<Option<Vec<T>>>, std::slice::Iter<'_, Option<Vec<T>>>), E>()
}
/// Prospective request for the same worker with an empty destination. No
/// allocation, callbacks or source mutation occur during this inspection.
pub(crate) fn reserve_required_bytes<T, E>(capacity: usize) -> Option<usize> {
    reserve_frame::<T, E>()?.checked_add(Layout::array::<T>(capacity).ok()?.size())
}
pub(crate) fn fixed_required_bytes<T: Copy, E>(source: &Vec<T>) -> Option<usize> {
    fixed_frame::<T, E>()?.checked_add(reserve_required_bytes::<T, E>(source.capacity())?)
}
pub(crate) fn vectors_required_bytes<T: Copy, E>(source: &Vec<Vec<T>>) -> Option<usize> {
    source.iter().try_fold(vectors_frame::<T, E>()?.checked_add(
        reserve_required_bytes::<Vec<T>, E>(source.capacity())?)?,
        |total, value| total.checked_add(fixed_required_bytes::<T, E>(value)?))
}
pub(crate) fn optional_vectors_required_bytes<T: Copy, E>(source: &Vec<Option<Vec<T>>>) -> Option<usize> {
    source.iter().try_fold(optional_vectors_frame::<T, E>()?.checked_add(
        reserve_required_bytes::<Option<Vec<T>>, E>(source.capacity())?)?,
        |total, value| match value {
            Some(value) => total.checked_add(fixed_required_bytes::<T, E>(value)?),
            None => Some(total),
        })
}
pub(crate) fn reserve<T, F: Fn(usize) -> Result<(), E>, E>(
    target: &mut Vec<T>, capacity: usize, funding: &F,
) -> Result<(), Error<E>> {
    funding(reserve_frame::<T, E>().ok_or(Error::Overflow)?).map_err(Error::Funding)?;
    if capacity > target.capacity() {
        let bytes = Layout::array::<T>(capacity).map_err(|_| Error::Overflow)?.size();
        funding(bytes).map_err(Error::Funding)?;
        target.try_reserve_exact(capacity - target.len()).map_err(Error::Allocation)?;
        if target.capacity() != capacity { return Err(Error::Capacity); }
    }
    Ok(())
}
pub(crate) fn fixed<T: Copy, F: Fn(usize) -> Result<(), E>, E>(
    target: &mut Vec<T>, source: &Vec<T>, funding: &F,
) -> Result<(), Error<E>> {
    funding(fixed_frame::<T, E>().ok_or(Error::Overflow)?).map_err(Error::Funding)?;
    reserve(target, source.capacity(), funding)?;
    target.clear();
    target.extend_from_slice(source);
    Ok(())
}
pub(crate) fn vectors<T: Copy, F: Fn(usize) -> Result<(), E>, E>(
    target: &mut Vec<Vec<T>>, source: &Vec<Vec<T>>, funding: &F,
) -> Result<(), Error<E>> {
    funding(vectors_frame::<T, E>().ok_or(Error::Overflow)?).map_err(Error::Funding)?;
    reserve(target, source.capacity(), funding)?;
    target.clear();
    for value in source {
        target.push(Vec::new());
        fixed(target.last_mut().expect("copy child"), value, funding)?;
    }
    Ok(())
}
pub(crate) fn optional_vectors<T: Copy, F: Fn(usize) -> Result<(), E>, E>(
    target: &mut Vec<Option<Vec<T>>>, source: &Vec<Option<Vec<T>>>, funding: &F,
) -> Result<(), Error<E>> {
    funding(optional_vectors_frame::<T, E>().ok_or(Error::Overflow)?).map_err(Error::Funding)?;
    reserve(target, source.capacity(), funding)?;
    target.clear();
    for value in source {
        target.push(value.as_ref().map(|_| Vec::new()));
        if let Some(value) = value {
            fixed(target.last_mut().expect("copy slot").as_mut().expect("copy child"), value, funding)?;
        }
    }
    Ok(())
}

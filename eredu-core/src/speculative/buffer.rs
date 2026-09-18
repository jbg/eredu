//! Fixed host destinations for the shared speculative driver. Custody proves
//! lifetime only; the native caller must admit each actual birth separately.
use crate::{GenerationError, HostPreparationAuthority};
use std::{
    fmt,
    iter::FusedIterator,
    ops::{Deref, DerefMut},
};

/// Mutable driver storage whose retained mode never grows or exports a Vec.
/// Ordinary Vec inputs keep ordinary allocation behavior. There is no operation
/// that attaches managed custody to an already-created ordinary buffer.
pub struct SpeculativeBuffer<T> {
    values: Vec<T>,
    limit: Option<usize>,
    // All elements and the allocation retire before their paying host token.
    authority: HostPreparationAuthority,
}
#[derive(Debug, thiserror::Error)]
enum AllocationCause {
    #[error("speculative host destination allocation failed: {0}")]
    Reserve(#[source] std::collections::TryReserveError),
    #[error("speculative host destination capacity differs from its request")]
    Capacity,
}
/// Allocation refusal retains the host token through its exact typed cause.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct SpeculativeBufferAllocationError {
    #[source]
    cause: AllocationCause,
    authority: HostPreparationAuthority,
}
impl<T> SpeculativeBuffer<T> {
    pub(super) fn is_funded_by(&self, funding: &crate::HostMetadataFunding) -> bool {
        self.authority.is_funded_by(funding)
    }
    /// Ordinary allocation, with the same growth behavior as Vec.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            limit: None,
            authority: HostPreparationAuthority::unmanaged(),
        }
    }
    /// Actual requested buffer and fixed constructor/iterator/error controls.
    /// Authority construction and outer error transport are separate producers.
    pub fn retained_control_bytes(capacity: usize) -> Option<usize> {
        use std::{alloc::Layout, mem::size_of};
        let parts = [
            Layout::array::<T>(capacity).ok()?.size(),
            size_of::<Self>(),
            size_of::<Vec<T>>(),
            size_of::<Result<Self, SpeculativeBufferAllocationError>>(),
            size_of::<SpeculativeBufferAllocationError>(),
            size_of::<AllocationCause>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<(usize, HostPreparationAuthority)>(),
            size_of::<SpeculativeBufferIntoIter<T>>(),
            size_of::<Option<T>>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    /// Constructs this destination after its exact request was admitted. The
    /// allocation happens here; no ordinary buffer can be relabelled retained.
    /// Failure destroys any allocation before returning its still-live custody.
    pub fn try_new_retained(
        capacity: usize,
        authority: HostPreparationAuthority,
    ) -> Result<Self, SpeculativeBufferAllocationError> {
        let mut values = Vec::new();
        if let Err(cause) = values.try_reserve_exact(capacity) {
            drop(values);
            return Err(SpeculativeBufferAllocationError {
                cause: AllocationCause::Reserve(cause),
                authority,
            });
        }
        if std::mem::size_of::<T>() != 0 && values.capacity() != capacity {
            drop(values);
            return Err(SpeculativeBufferAllocationError {
                cause: AllocationCause::Capacity,
                authority,
            });
        }
        Ok(Self {
            values,
            limit: Some(capacity),
            authority,
        })
    }
    /// Adds one value without growing a retained destination.
    pub fn try_push(&mut self, value: T) -> Result<(), GenerationError> {
        if self.limit.is_some_and(|limit| self.values.len() == limit) {
            return Err(GenerationError::InvalidStorage);
        }
        self.values.push(value);
        Ok(())
    }
    /// Moves values into the existing destination. Failure never reallocates a
    /// retained buffer; already moved values stay owned by this buffer.
    pub fn try_extend(
        &mut self,
        values: impl IntoIterator<Item = T>,
    ) -> Result<(), GenerationError> {
        for value in values {
            self.try_push(value)?;
        }
        Ok(())
    }
    /// Ordinary-only export for legacy mechanism adapters. Retained storage
    /// returns unchanged, with its complete authority and allocation still held.
    pub fn try_into_ordinary(self) -> Result<Vec<T>, Self> {
        if self.limit.is_some() { Err(self) } else { Ok(self.values) }
    }
    /// Removes one already-owned value without constructing or growing storage.
    pub fn pop(&mut self) -> Option<T> { self.values.pop() }
    /// Drains all rows while borrowing the fixed allocation and its custody.
    pub fn drain(&mut self)->std::vec::Drain<'_,T> { self.values.drain(..) }
    /// Destroys values while retaining this allocation and its custody.
    pub fn clear(&mut self) { self.values.clear(); }
    /// Actual admitted element capacity; retained storage never grows beyond it.
    pub fn capacity(&self) -> usize {
        self.limit.unwrap_or(self.values.capacity())
    }
    /// Removes one row in place without allocating or changing its custody.
    pub fn remove(&mut self, index: usize) -> Option<T> {
        (index < self.values.len()).then(|| self.values.remove(index))
    }
    /// Discards an already-consumed prefix in place, preserving capacity/custody.
    pub(crate) fn remove_first(&mut self) -> Option<T> {
        if self.values.is_empty() {
            None
        } else {
            Some(self.values.remove(0))
        }
    }
}
impl<T> Default for SpeculativeBuffer<T> {
    fn default() -> Self {
        Vec::new().into()
    }
}
impl<T> From<Vec<T>> for SpeculativeBuffer<T> {
    fn from(values: Vec<T>) -> Self {
        Self {
            values,
            limit: None,
            authority: HostPreparationAuthority::unmanaged(),
        }
    }
}
impl<T> Deref for SpeculativeBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.values
    }
}
impl<T> DerefMut for SpeculativeBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
}
impl<T: fmt::Debug> fmt::Debug for SpeculativeBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.values.fmt(f)
    }
}
impl<T: PartialEq> PartialEq for SpeculativeBuffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}
impl<T: Eq> Eq for SpeculativeBuffer<T> {}
impl<T: PartialEq, const N: usize> PartialEq<[T; N]> for SpeculativeBuffer<T> {
    fn eq(&self, other: &[T; N]) -> bool {
        self.values.as_slice() == other
    }
}
impl<T> AsRef<[T]> for SpeculativeBuffer<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}
impl<T> AsMut<[T]> for SpeculativeBuffer<T> {
    fn as_mut(&mut self) -> &mut [T] { self }
}
impl<'a, T> IntoIterator for &'a SpeculativeBuffer<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<'a, T> IntoIterator for &'a mut SpeculativeBuffer<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}
/// Owning iteration retains custody until all elements and the Vec allocation
/// retire, including early drop. It exposes no extraction of the underlying Vec.
pub struct SpeculativeBufferIntoIter<T> {
    values: std::vec::IntoIter<T>,
    authority: HostPreparationAuthority,
}
impl<T> fmt::Debug for SpeculativeBufferIntoIter<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpeculativeBufferIntoIter")
            .field("remaining", &self.values.len())
            .finish_non_exhaustive()
    }
}
impl<T> IntoIterator for SpeculativeBuffer<T> {
    type Item = T;
    type IntoIter = SpeculativeBufferIntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            values: self.values.into_iter(),
            authority: self.authority,
        }
    }
}
impl<T> Iterator for SpeculativeBufferIntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.values.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.values.size_hint()
    }
}
impl<T> DoubleEndedIterator for SpeculativeBufferIntoIter<T> {
    fn next_back(&mut self) -> Option<T> {
        self.values.next_back()
    }
}
impl<T> ExactSizeIterator for SpeculativeBufferIntoIter<T> {}
impl<T> FusedIterator for SpeculativeBufferIntoIter<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    };
    struct Custody(Arc<AtomicBool>);
    impl Drop for Custody {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    struct Item {
        retired: Arc<AtomicBool>,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for Item {
        fn drop(&mut self) {
            assert!(!self.retired.load(Ordering::SeqCst));
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn fixed_capacity_and_partial_iteration_keep_custody_through_payload_retirement() {
        let retired = Arc::new(AtomicBool::new(false));
        let drops = Arc::new(AtomicUsize::new(0));
        let authority = HostPreparationAuthority::retain(Custody(retired.clone()));
        let mut values = SpeculativeBuffer::try_new_retained(2, authority).unwrap();
        let item = || Item {
            retired: retired.clone(),
            drops: drops.clone(),
        };
        values.try_push(item()).unwrap();
        values.try_push(item()).unwrap();
        assert!(matches!(
            values.try_push(item()),
            Err(GenerationError::InvalidStorage)
        ));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(values.len(), 2);
        let mut iter = values.into_iter();
        drop(iter.next());
        assert!(!retired.load(Ordering::SeqCst));
        drop(iter);
        assert_eq!(drops.load(Ordering::SeqCst), 3);
        assert!(retired.load(Ordering::SeqCst));

        let refused = Arc::new(AtomicBool::new(false));
        let error = SpeculativeBuffer::<u64>::try_new_retained(
            usize::MAX,
            HostPreparationAuthority::retain(Custody(refused.clone())),
        )
        .unwrap_err();
        assert!(std::error::Error::source(&error).is_some());
        assert!(!refused.load(Ordering::SeqCst));
        drop(error);
        assert!(refused.load(Ordering::SeqCst));
    }
}

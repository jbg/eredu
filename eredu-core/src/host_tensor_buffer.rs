//! Retained host tensor values and their allocation-accounting custody.
use std::{
    fmt,
    iter::FusedIterator,
    ops::{Deref, DerefMut},
};

trait ErasedCustody: Send + Sync {
    fn retire(self: Box<Self>);
}
fn take_box<C>(owner: Box<C>) -> C {
    *owner
}
impl<C: Send + Sync> ErasedCustody for C {
    fn retire(self: Box<Self>) {
        // Return the value only after its paid box allocation has retired.
        drop(take_box(self));
    }
}
struct Custody(Option<Box<dyn ErasedCustody>>);
impl Drop for Custody {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            owner.retire();
        }
    }
}

/// Host tensor storage whose accounting owner outlives the backing allocation.
///
/// A producer admits the vector's full capacity and the custody box before
/// constructing them. This owner grants no execution or allocation authority.
/// Reading, mutating, truncating, and consuming values never transfers the
/// backing allocation away from its accounting custody.
pub struct HostTensorBuffer<T> {
    // Declaration order also governs unwinding: backing retires before custody.
    values: Vec<T>,
    custody: Custody,
}
impl<T> HostTensorBuffer<T> {
    /// Retains already admitted backing and its actual allocation owner.
    /// The producer funds [`Self::custody_allocation_bytes`] before this call.
    /// A caller-owned portable implementation may use a zero-sized owner.
    pub fn new<C: Send + Sync + 'static>(values: Vec<T>, custody: C) -> Self {
        Self {
            values,
            custody: Custody(Some(Box::new(custody))),
        }
    }
    /// Requested heap allocation for the concrete erased custody owner. The
    /// containing buffer's inline representation is quoted by its producer.
    pub const fn custody_allocation_bytes<C: Send + Sync + 'static>() -> usize {
        std::mem::size_of::<C>()
    }
    /// Full retained vector capacity, independent of the initialized length.
    pub fn capacity(&self) -> usize {
        self.values.capacity()
    }
    /// Borrow initialized values while their backing remains owned.
    pub fn as_slice(&self) -> &[T] {
        &self.values
    }
    /// Borrow initialized values mutably without growing their allocation.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.values
    }
    /// Discard a suffix while preserving the full backing charge.
    pub fn truncate(&mut self, len: usize) {
        self.values.truncate(len);
    }
}
impl<T> Deref for HostTensorBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.values
    }
}
impl<T> DerefMut for HostTensorBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
}
impl<T> AsRef<[T]> for HostTensorBuffer<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}
impl<T> AsMut<[T]> for HostTensorBuffer<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}
impl<T: fmt::Debug> fmt::Debug for HostTensorBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.values.fmt(f)
    }
}
impl<T: PartialEq, R: AsRef<[T]>> PartialEq<R> for HostTensorBuffer<T> {
    fn eq(&self, other: &R) -> bool {
        self.as_slice() == other.as_ref()
    }
}
impl<T: Eq> Eq for HostTensorBuffer<T> {}

/// Consuming iterator retaining the original backing charge through exhaustion
/// or early drop. Yielded scalar values do not carry the backing allocation.
pub struct HostTensorBufferIntoIter<T> {
    values: std::vec::IntoIter<T>,
    _custody: Custody,
}
impl<T> Iterator for HostTensorBufferIntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.values.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.values.size_hint()
    }
    fn nth(&mut self, n: usize) -> Option<T> {
        self.values.nth(n)
    }
}
impl<T> DoubleEndedIterator for HostTensorBufferIntoIter<T> {
    fn next_back(&mut self) -> Option<T> {
        self.values.next_back()
    }
    fn nth_back(&mut self, n: usize) -> Option<T> {
        self.values.nth_back(n)
    }
}
impl<T> ExactSizeIterator for HostTensorBufferIntoIter<T> {
    fn len(&self) -> usize {
        self.values.len()
    }
}
impl<T> FusedIterator for HostTensorBufferIntoIter<T> {}
impl<T> IntoIterator for HostTensorBuffer<T> {
    type Item = T;
    type IntoIter = HostTensorBufferIntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        let Self { values, custody } = self;
        HostTensorBufferIntoIter {
            values: values.into_iter(),
            _custody: custody,
        }
    }
}
impl<'a, T> IntoIterator for &'a HostTensorBuffer<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<'a, T> IntoIterator for &'a mut HostTensorBuffer<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    struct Event(&'static str, Arc<Mutex<Vec<&'static str>>>);
    impl Drop for Event {
        fn drop(&mut self) {
            self.1.lock().unwrap().push(self.0);
        }
    }
    fn fixture() -> (HostTensorBuffer<Event>, Arc<Mutex<Vec<&'static str>>>) {
        let events = Arc::new(Mutex::new(Vec::new()));
        (
            HostTensorBuffer::new(
                vec![
                    Event("first", events.clone()),
                    Event("second", events.clone()),
                ],
                Event("custody", events.clone()),
            ),
            events,
        )
    }
    #[test]
    fn backing_values_retire_before_custody() {
        let (buffer, events) = fixture();
        drop(buffer);
        assert_eq!(*events.lock().unwrap(), ["first", "second", "custody"]);
    }
    #[test]
    fn partial_iteration_retains_original_capacity_until_iterator_drop() {
        let (buffer, events) = fixture();
        let mut values = buffer.into_iter();
        let first = values.next().unwrap();
        assert!(events.lock().unwrap().is_empty());
        assert_eq!(values.len(), 1);
        drop(values);
        assert_eq!(*events.lock().unwrap(), ["second", "custody"]);
        drop(first);
    }
    #[test]
    fn exhaustion_retains_custody_until_iterator_retires() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut values =
            HostTensorBuffer::new(vec![3, 5], Event("custody", events.clone())).into_iter();
        assert_eq!(values.next_back(), Some(5));
        assert_eq!(values.next(), Some(3));
        assert_eq!(values.next(), None);
        assert!(events.lock().unwrap().is_empty());
        drop(values);
        assert_eq!(*events.lock().unwrap(), ["custody"]);
    }
    #[test]
    fn truncation_and_mutation_preserve_backing_capacity_and_custody() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut buffer = HostTensorBuffer::new(vec![2, 7, 11], Event("custody", events.clone()));
        let capacity = buffer.capacity();
        buffer[0] = 3;
        buffer.truncate(1);
        assert_eq!(buffer, vec![3]);
        assert_eq!(buffer.capacity(), capacity);
        assert!(events.lock().unwrap().is_empty());
        drop(buffer);
        assert_eq!(*events.lock().unwrap(), ["custody"]);
    }
    #[test]
    fn unwinding_payload_retirement_still_precedes_custody() {
        struct Payload(Event, bool);
        impl Drop for Payload {
            fn drop(&mut self) {
                if self.1 {
                    panic!("payload retirement");
                }
            }
        }
        let events = Arc::new(Mutex::new(Vec::new()));
        let buffer = HostTensorBuffer::new(
            vec![
                Payload(Event("first", events.clone()), true),
                Payload(Event("second", events.clone()), false),
            ],
            Event("custody", events.clone()),
        );
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(buffer))).is_err());
        assert_eq!(*events.lock().unwrap(), ["first", "second", "custody"]);
    }
}

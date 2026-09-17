//! Immutable host record rows with closed ownership and unchanged wire arrays.
use super::{SpeculativeBuffer, SpeculativeDriverError, SpeculativeExecutor};
use crate::HostPreparationAuthority;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    alloc::Layout,
    fmt,
    mem::size_of,
    ops::Deref,
    sync::{atomic::AtomicUsize, Arc},
};

struct Payload<T> {
    values: SpeculativeBuffer<T>,
    _host: HostPreparationAuthority,
}
struct Owner<T>(Option<Arc<Payload<T>>>);
impl<T> Owner<T> {
    fn get(&self) -> &Payload<T> {
        self.0.as_deref().expect("live record owner")
    }
}
impl<T> Clone for Owner<T> {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(
            self.0.as_ref().expect("live record owner"),
        )))
    }
}
impl<T> Drop for Owner<T> {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
enum Storage<T> {
    Ordinary(Vec<T>),
    Retained(Owner<T>),
}

/// Immutable values in a controlled speculative record. Wire serialization is
/// the same array as Vec. Ordinary clones retain ordinary behavior; retained
/// clones share the exact existing allocation without cloning any row payload.
/// No raw Arc, Weak, mutable slice or retained Vec export is available.
pub struct SpeculativeValues<T>(Storage<T>);
impl<T> SpeculativeValues<T> {
    /// Shared shell and fixed constructor/retirement controls, excluding the
    /// separately allocated buffer and the caller's actual authority producer.
    pub fn retained_control_bytes() -> Option<usize> {
        let shared = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Payload<T>>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        [
            shared,
            size_of::<Self>(),
            size_of::<Storage<T>>(),
            size_of::<Payload<T>>(),
            size_of::<Option<Payload<T>>>(),
            size_of::<Owner<T>>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<Result<Vec<T>, SpeculativeBuffer<T>>>(),
            size_of::<(Self, usize, usize)>(),
            size_of::<(SpeculativeBuffer<T>, HostPreparationAuthority)>(),
            size_of::<std::vec::IntoIter<T>>(),
            size_of::<Option<T>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    /// Freezes a freshly produced driver buffer after admitting this exact
    /// shared shell plus caller-supplied enclosing record controls. An ordinary
    /// buffer is kept ordinary and never relabelled with the supplied custody.
    pub fn from_driver_buffer<E: SpeculativeExecutor>(
        values: SpeculativeBuffer<T>,
        executor: &E,
        context: E::Context<'_>,
        enclosing_controls: usize,
    ) -> Result<Self, SpeculativeDriverError<E::Error>> {
        match values.try_into_ordinary() {
            Ok(values) => Ok(Self::from(values)),
            Err(values) => {
                let controls = Self::retained_control_bytes()
                    .and_then(|n| n.checked_add(enclosing_controls))
                    .and_then(|n| {
                        n.checked_add(size_of::<Result<Self, SpeculativeDriverError<E::Error>>>())
                    });
                let host = executor.driver_host_metadata(controls, context)?;
                Ok(Self::from_prepared_buffer(values, host))
            }
        }
    }
    /// Copies/moves exactly the supplied rows into a newly paid destination,
    /// then freezes it. Source allocations never become retained destinations.
    pub fn collect_with_metadata<E: SpeculativeExecutor>(
        values: impl ExactSizeIterator<Item = T>,
        executor: &E,
        context: E::Context<'_>,
        enclosing_controls: usize,
    ) -> Result<Self, SpeculativeDriverError<E::Error>> {
        let mut destination = executor.driver_buffer(values.len(), context)?;
        destination
            .try_extend(values)
            .map_err(SpeculativeDriverError::Generation)?;
        Self::from_driver_buffer(destination, executor, context, enclosing_controls)
    }
    /// Freezes a freshly admitted buffer after the caller pays
    /// `retained_control_bytes` and the actual host-authority producer. Ordinary
    /// buffers keep their ordinary storage and cloning behavior.
    pub fn from_prepared_buffer(values: SpeculativeBuffer<T>, host: HostPreparationAuthority) -> Self {
        match values.try_into_ordinary() {
            Ok(values) => Self::from(values),
            Err(values) => Self(Storage::Retained(Owner(Some(Arc::new(Payload { values, _host: host }))))),
        }
    }
    /// Borrowed immutable wire values.
    pub fn as_slice(&self) -> &[T] {
        match &self.0 {
            Storage::Ordinary(v) => v,
            Storage::Retained(v) => &v.get().values,
        }
    }
    /// Whether this wire array has no values.
    pub fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }
}
impl<T> Default for SpeculativeValues<T> {
    fn default() -> Self {
        Vec::new().into()
    }
}
impl<T> From<Vec<T>> for SpeculativeValues<T> {
    fn from(v: Vec<T>) -> Self {
        Self(Storage::Ordinary(v))
    }
}
impl<T: Clone> Clone for SpeculativeValues<T> {
    fn clone(&self) -> Self {
        match &self.0 {
            Storage::Ordinary(v) => v.clone().into(),
            Storage::Retained(v) => Self(Storage::Retained(v.clone())),
        }
    }
}
impl<T> Deref for SpeculativeValues<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T> AsRef<[T]> for SpeculativeValues<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T: fmt::Debug> fmt::Debug for SpeculativeValues<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}
impl<T: PartialEq> PartialEq for SpeculativeValues<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl<T: Eq> Eq for SpeculativeValues<T> {}
impl<T: PartialEq> PartialEq<Vec<T>> for SpeculativeValues<T> {
    fn eq(&self, other: &Vec<T>) -> bool {
        self.as_slice() == other
    }
}
impl<T: PartialEq, const N: usize> PartialEq<[T; N]> for SpeculativeValues<T> {
    fn eq(&self, other: &[T; N]) -> bool {
        self.as_slice() == other
    }
}
impl<T: Serialize> Serialize for SpeculativeValues<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.as_slice().serialize(s)
    }
}
impl<'de, T: Deserialize<'de>> Deserialize<'de> for SpeculativeValues<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Vec::<T>::deserialize(d).map(Self::from)
    }
}
impl<'a, T> IntoIterator for &'a SpeculativeValues<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

enum Iter<T: Copy> {
    Ordinary(std::vec::IntoIter<T>),
    Retained {
        source: SpeculativeValues<T>,
        start: usize,
        end: usize,
    },
}
/// Owning iteration over inline Copy rows retains the full source until the
/// iterator retires. Non-Copy payloads remain available through borrowed rows.
pub struct SpeculativeValuesIntoIter<T: Copy>(Iter<T>);
impl<T: Copy> fmt::Debug for SpeculativeValuesIntoIter<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpeculativeValuesIntoIter")
            .field("remaining", &self.len())
            .finish()
    }
}
impl<T: Copy> Iterator for SpeculativeValuesIntoIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        match &mut self.0 {
            Iter::Ordinary(i) => i.next(),
            Iter::Retained { source, start, end } => {
                if *start == *end {
                    None
                } else {
                    let value = source[*start];
                    *start += 1;
                    Some(value)
                }
            }
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match &self.0 {
            Iter::Ordinary(i) => i.len(),
            Iter::Retained { start, end, .. } => end - start,
        };
        (n, Some(n))
    }
}
impl<T: Copy> DoubleEndedIterator for SpeculativeValuesIntoIter<T> {
    fn next_back(&mut self) -> Option<T> {
        match &mut self.0 {
            Iter::Ordinary(i) => i.next_back(),
            Iter::Retained { source, start, end } => {
                if *start == *end {
                    None
                } else {
                    *end -= 1;
                    Some(source[*end])
                }
            }
        }
    }
}
impl<T: Copy> ExactSizeIterator for SpeculativeValuesIntoIter<T> {}
impl<T: Copy> std::iter::FusedIterator for SpeculativeValuesIntoIter<T> {}
impl<T: Copy> IntoIterator for SpeculativeValues<T> {
    type Item = T;
    type IntoIter = SpeculativeValuesIntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        SpeculativeValuesIntoIter(match self.0 {
            Storage::Ordinary(v) => Iter::Ordinary(v.into_iter()),
            storage => {
                let source = Self(storage);
                let end = source.len();
                Iter::Retained {
                    source,
                    start: 0,
                    end,
                }
            }
        })
    }
}

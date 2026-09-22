//! Exact managed host extents for ordinary registry transactions.
use super::*;
use crate::working_memory::qualified_storage;

pub(super) fn vector_bytes<T>(capacity: usize) -> Result<u64, WorkingMemoryError> {
    qualified_storage::array_bytes::<T>(capacity)
}
pub(super) fn registration_bytes<K: Ord + Send + 'static>(
    capacity: usize,
) -> Result<u64, WorkingMemoryError> {
    qualified_storage::shared_bytes::<Registration<K>>()?
        .checked_add(vector_bytes::<K>(capacity)?)
        .ok_or(WorkingMemoryError::Overflow)
}
/// Independently retained registrations and their exact ordered-container charge.
/// Removing a registration does not release the container's backing allocation.
#[derive(Debug)]
pub struct StorageRegistrations<K: Ord + Send + 'static> {
    entries: Vec<(K, WorkingMemoryStorage<K>)>,
    preparation: Option<eredu_core::HostPreparationAuthority>,
}
impl<K: Ord + Send + 'static> StorageRegistrations<K> {
    pub(super) fn pending(
        entries: Vec<(K, WorkingMemoryStorage<K>)>,
    ) -> Result<Self, WorkingMemoryError> {
        Ok(Self {
            entries,
            preparation: None,
        })
    }
    pub(super) fn retain_preparation(&mut self, host: &eredu_core::HostPreparationAuthority) {
        self.preparation = Some(host.clone());
    }
    /// Exact retained container and registration extents for the stated population.
    /// Publication also needs a canonical namespace, batch and transaction rows.
    pub fn retained_control_bytes(population: usize) -> Result<u64, WorkingMemoryError> {
        let owners = registration_bytes::<K>(1)?
            .checked_mul(u64::try_from(population).map_err(|_| WorkingMemoryError::Overflow)?)
            .ok_or(WorkingMemoryError::Overflow)?;
        owners
            .checked_add(vector_bytes::<(K, WorkingMemoryStorage<K>)>(population)?)
            .ok_or(WorkingMemoryError::Overflow)
    }
    /// Number of retained independent handles.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// Whether every handle has been removed.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Borrows a handle without allocating or changing ownership.
    pub fn get(&self, key: &K) -> Option<&WorkingMemoryStorage<K>> {
        self.entries
            .binary_search_by(|(candidate, _)| candidate.cmp(key))
            .ok()
            .map(|i| &self.entries[i].1)
    }
    /// Moves one handle out; the ordered container keeps its charge.
    pub fn remove(&mut self, key: &K) -> Option<WorkingMemoryStorage<K>> {
        let index = self
            .entries
            .binary_search_by(|(candidate, _)| candidate.cmp(key))
            .ok()?;
        let (key, value) = self.entries.remove(index);
        drop(key);
        Some(value)
    }
    /// Borrows the independent registrations in key order.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &WorkingMemoryStorage<K>> {
        self.entries.iter().map(|(_, value)| value)
    }
    pub(super) fn values_mut(&mut self) -> impl Iterator<Item = &mut WorkingMemoryStorage<K>> {
        self.entries.iter_mut().map(|(_, value)| value)
    }
    /// Moves handles in key order while retaining the container's charge until
    /// its storage and remaining keys have retired.
    pub fn into_values(self) -> StorageRegistrationValues<K> {
        StorageRegistrationValues {
            entries: self.entries.into_iter(),
            _preparation: self.preparation,
        }
    }
}
/// Owning iterator whose final field retains the consumed container's charge.
pub struct StorageRegistrationValues<K: Ord + Send + 'static> {
    entries: std::vec::IntoIter<(K, WorkingMemoryStorage<K>)>,
    _preparation: Option<eredu_core::HostPreparationAuthority>,
}
impl<K: Ord + Send + 'static> Iterator for StorageRegistrationValues<K> {
    type Item = WorkingMemoryStorage<K>;
    fn next(&mut self) -> Option<Self::Item> {
        self.entries.next().map(|(key, value)| {
            drop(key);
            value
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.entries.size_hint()
    }
}
impl<K: Ord + Send + 'static> ExactSizeIterator for StorageRegistrationValues<K> {}

impl<K: Ord + Send + 'static> std::ops::Index<&K> for StorageRegistrations<K> {
    type Output = WorkingMemoryStorage<K>;
    fn index(&self, key: &K) -> &Self::Output {
        self.get(key).expect("registered key")
    }
}

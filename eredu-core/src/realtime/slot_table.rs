//! Canonical delayed-coordinate records with explicit contiguous storage.
use super::RealtimeSlotCoordinate;
use crate::{HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};

/// Ordered realtime records with a concrete contiguous host allocation layout.
#[derive(Debug)]
pub struct RealtimeSlotTable<V> {
    entries: Vec<(RealtimeSlotCoordinate, V)>,
    maximum: Option<usize>,
}
impl<V: Clone> Clone for RealtimeSlotTable<V> {
    fn clone(&self) -> Self {
        let mut entries = Vec::with_capacity(self.maximum.unwrap_or(self.entries.len()));
        entries.extend(self.entries.iter().cloned());
        Self {
            entries,
            maximum: self.maximum,
        }
    }
}
impl<V: PartialEq> PartialEq for RealtimeSlotTable<V> {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries
    }
}
impl<V: Eq> Eq for RealtimeSlotTable<V> {}
impl<V> Default for RealtimeSlotTable<V> {
    fn default() -> Self {
        Self::new()
    }
}
impl<V> RealtimeSlotTable<V> {
    /// Empty ordinary canonical table.
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            maximum: None,
        }
    }
    /// Whether destination capacity belongs to an explicit paid source.
    pub fn is_prepared(&self)->bool {self.maximum.is_some()}
    /// Current retained coordinate count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    /// Whether no coordinate is retained.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Requested contiguous backing; dynamic children of V are separate.
    pub fn backing_bytes(capacity: usize) -> Option<usize> {
        std::alloc::Layout::array::<(RealtimeSlotCoordinate, V)>(capacity)
            .ok()
            .map(|v| v.size())
    }
    /// Fixed constructor frames plus the requested destination backing.
    pub fn construction_bytes(capacity: usize) -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Vec<(RealtimeSlotCoordinate, V)>>(),
            size_of::<Result<Self, HostMetadataFundingError>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<(usize, usize)>(),
            size_of::<Option<&HostMetadataFunding>>(),
            HostMetadataFunding::reservation_control_bytes(),
        ];
        frames.into_iter().try_fold(
            size_of_val(&frames).checked_add(Self::backing_bytes(capacity)?)?,
            usize::checked_add,
        )
    }
    /// Borrow a value at its exact semantic coordinate.
    pub fn get(&self, key: &RealtimeSlotCoordinate) -> Option<&V> {
        self.entries
            .binary_search_by_key(key, |row| row.0)
            .ok()
            .map(|i| &self.entries[i].1)
    }
    /// Whether an exact coordinate is retained.
    pub fn contains_key(&self, key: &RealtimeSlotCoordinate) -> bool {
        self.get(key).is_some()
    }
    /// Stable coordinate-ordered records.
    pub fn iter(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&RealtimeSlotCoordinate, &V)> + ExactSizeIterator {
        self.entries.iter().map(|(key, value)| (key, value))
    }
    /// Stable coordinate-ordered values.
    pub fn values(&self) -> impl DoubleEndedIterator<Item = &V> + ExactSizeIterator {
        self.entries.iter().map(|(_, value)| value)
    }
    /// Ordinary insertion with exact incremental growth and replacement.
    /// Paid destinations use insert_prepared, which returns capacity refusal.
    pub fn insert(&mut self, key: RealtimeSlotCoordinate, value: V) -> Option<V> {
        assert!(
            self.maximum.is_none(),
            "prepared realtime table requires checked insertion"
        );
        match self.entries.binary_search_by_key(&key, |row| row.0) {
            Ok(i) => Some(std::mem::replace(&mut self.entries[i].1, value)),
            Err(i) => {
                if self.entries.len() == self.entries.capacity() {
                    self.entries.reserve_exact(1);
                }
                self.entries.insert(i, (key, value));
                None
            }
        }
    }
    /// Uses the same insertion policy for ordinary and prepared directories.
    /// Prepared storage never grows beyond its already paid destination.
    pub fn insert_checked(&mut self,key:RealtimeSlotCoordinate,value:V)
        ->Result<Option<V>,(HostMetadataFundingError,V)> {
        if self.maximum.is_some() {self.insert_prepared(key,value)}
        else {Ok(self.insert(key,value))}
    }
    /// Retains records accepted by the existing coordinate policy.
    pub fn retain(&mut self, mut keep: impl FnMut(&RealtimeSlotCoordinate, &mut V) -> bool) {
        self.entries.retain_mut(|(key, value)| keep(key, value));
    }
    /// Creates a paid fixed destination before cloning values through the
    /// caller's source-qualified payload worker. The enclosing state retains
    /// funding until both successful and failed transaction storage retires.
    pub fn try_clone_with<E>(
        &self, additional: usize, funding: &HostMetadataFunding,
        clone_value: impl FnMut(&V)->Result<V,E>,
    ) -> Result<Self,E> where E: From<HostMetadataFundingError> {
        self.try_map_with(additional, funding, clone_value)
    }
    /// Actual mapping frame and destination extent, excluding the mapper's
    /// dynamic child allocations. Querying this creates no destination.
    pub fn map_control_bytes<Q,E,F>(&self,additional:usize,_map:&F)->Option<usize> {
        Self::map_capacity_control_bytes::<Q,E,F>(self.len().checked_add(additional)?,_map)
    }
    /// Exact mapping frames at a source-declared destination capacity.
    pub fn map_capacity_control_bytes<Q,E,F>(capacity:usize,_map:&F)->Option<usize> {
        let frames=[size_of::<F>(),size_of::<E>(),size_of::<Result<Q,E>>(),
            size_of::<Result<RealtimeSlotTable<Q>,E>>(),size_of::<(&Self,&HostMetadataFunding)>()];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)?
            .checked_add(RealtimeSlotTable::<Q>::construction_bytes(capacity)?)
    }
    /// Maps the actual ordered records into a paid destination while preserving
    /// every semantic coordinate. The mapper owns its payload child source.
    pub fn try_map_with<'a,Q,E,F>(
        &'a self, additional: usize, funding: &HostMetadataFunding, mut map: F,
    ) -> Result<RealtimeSlotTable<Q>,E>
    where E: From<HostMetadataFundingError>, F: FnMut(&'a V)->Result<Q,E> {
        let capacity=self.len().checked_add(additional).ok_or(HostMetadataFundingError::Overflow)?;
        let controls=self.map_control_bytes::<Q,E,F>(additional,&map)
            .ok_or(HostMetadataFundingError::Overflow)?;
        funding.reserve_metadata(controls)?;
        let mut entries=Vec::new();
        entries.try_reserve_exact(capacity).map_err(|_|HostMetadataFundingError::Unavailable)?;
        for (key,value) in &self.entries {entries.push((*key,map(value)?));}
        Ok(RealtimeSlotTable{entries,maximum:Some(capacity)})
    }
    /// Replaces only directory backing under the supplied account. Values move
    /// without cloning, and refusal leaves all old records unchanged.
    pub fn reserve_prepared(&mut self,additional:usize,funding:&HostMetadataFunding)
        ->Result<(),HostMetadataFundingError> {
        let capacity=self.len().checked_add(additional).ok_or(HostMetadataFundingError::Overflow)?;
        funding.reserve_metadata(Self::construction_bytes(capacity).ok_or(HostMetadataFundingError::Overflow)?)?;
        let mut entries=Vec::new();
        entries.try_reserve_exact(capacity).map_err(|_|HostMetadataFundingError::Unavailable)?;
        entries.append(&mut self.entries);
        self.entries=entries;self.maximum=Some(capacity);Ok(())
    }
    /// Checked insertion independent of allocator overcapacity. Refusal returns
    /// the unchanged value before mutation.
    pub fn insert_prepared(
        &mut self,
        key: RealtimeSlotCoordinate,
        value: V,
    ) -> Result<Option<V>, (HostMetadataFundingError, V)> {
        let Some(maximum) = self.maximum else {
            return Err((HostMetadataFundingError::Unavailable, value));
        };
        match self.entries.binary_search_by_key(&key, |row| row.0) {
            Ok(i) => Ok(Some(std::mem::replace(&mut self.entries[i].1, value))),
            Err(i) => {
                if self.entries.len() == maximum {
                    return Err((HostMetadataFundingError::Unavailable, value));
                }
                self.entries.insert(i, (key, value));
                Ok(None)
            }
        }
    }
}
impl<V> Extend<(RealtimeSlotCoordinate, V)> for RealtimeSlotTable<V> {
    fn extend<I: IntoIterator<Item = (RealtimeSlotCoordinate, V)>>(&mut self, iter: I) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}
impl<V> IntoIterator for RealtimeSlotTable<V> {
    type Item = (RealtimeSlotCoordinate, V);
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

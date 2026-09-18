//! Ordered scratch keyed by the actual dense state index, not padded byte IDs.
use crate::util::{
    allocation::{AllocationError, Allocator},
    primitives::StateID,
};
use alloc::vec::Vec;

#[derive(Debug)]
pub(crate) struct StateMap<T> {
    rows: Vec<Option<T>>,
    stride2: usize,
    len: usize,
}
impl<T> StateMap<T> {
    pub(crate) fn new(stride2: usize) -> Self {
        Self {
            rows: Vec::new(),
            stride2,
            len: 0,
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.len
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub(crate) fn contains_key(&self, id: &StateID) -> bool {
        self.rows
            .get(id.as_usize() >> self.stride2)
            .is_some_and(Option::is_some)
    }
    pub(crate) fn insert(
        &mut self,
        id: StateID,
        value: T,
        allocation: Allocator<'_>,
    ) -> Result<Option<T>, AllocationError> {
        let index = id.as_usize() >> self.stride2;
        if index >= self.rows.len() {
            let additional =
                index.checked_add(1).ok_or(AllocationError::SizeOverflow)? - self.rows.len();
            allocation.grow(&mut self.rows, additional)?;
            self.rows.resize_with(index + 1, || None);
        }
        let old = self.rows[index].replace(value);
        if old.is_none() {
            self.len += 1;
        }
        Ok(old)
    }
    pub(crate) fn remove(&mut self, id: &StateID) -> Option<T> {
        let value = self.rows.get_mut(id.as_usize() >> self.stride2)?.take();
        if value.is_some() {
            self.len -= 1;
        }
        value
    }
    pub(crate) fn iter(&self) -> impl Iterator<Item = (StateID, &T)> {
        let stride2 = self.stride2;
        self.rows
            .iter()
            .enumerate()
            .filter_map(move |(index, value)| {
                value
                    .as_ref()
                    .map(|value| (StateID::new(index << stride2).unwrap(), value))
            })
    }
    pub(crate) fn into_iter(self) -> impl Iterator<Item = (StateID, T)> {
        let stride2 = self.stride2;
        self.rows
            .into_iter()
            .enumerate()
            .filter_map(move |(index, value)| {
                value.map(|value| (StateID::new(index << stride2).unwrap(), value))
            })
    }
}

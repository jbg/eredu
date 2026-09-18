//! Prospective storage admission shared by registry and URI construction.
//!
//! The caller retains its owner while the returned registry and derived values live.
use core::{alloc::Layout, fmt, hash::Hash};
pub use fluent_uri::allocation::{Allocation, AllocationError, Unenforced};
use std::{
    collections::VecDeque,
    sync::{atomic::AtomicUsize, Arc},
};

/// A borrowed allocation policy. It owns no heap storage.
#[derive(Clone, Copy)]
pub(crate) struct Allocator<'a>(pub(crate) &'a dyn Allocation);
impl fmt::Debug for Allocator<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Allocator")
    }
}
impl<'a> Allocator<'a> {
    pub(crate) fn copy_str(self, text: &str) -> Result<String, AllocationError> {
        let mut out = String::new();
        self.grow_string(&mut out, text.len())?;
        out.push_str(text);
        Ok(out)
    }
    pub(crate) fn grow_string(
        self,
        text: &mut String,
        additional: usize,
    ) -> Result<(), AllocationError> {
        let needed = text
            .len()
            .checked_add(additional)
            .ok_or(AllocationError::SizeOverflow)?;
        if needed <= text.capacity() {
            return Ok(());
        }
        self.layout(Layout::array::<u8>(needed).map_err(|_| AllocationError::SizeOverflow)?)?;
        text.try_reserve_exact(additional)
            .map_err(|_| AllocationError::HostAllocation)
    }
    pub(crate) fn layout(self, layout: Layout) -> Result<(), AllocationError> {
        if layout.size() == 0 {
            return Ok(());
        }
        self.0.reserve(layout.size())
    }
    pub(crate) fn arc<T>(self, value: T) -> Result<Arc<T>, AllocationError> {
        // Arc's allocation contains its two atomic reference counts followed by T.
        let (layout, _) = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<T>())
            .map_err(|_| AllocationError::SizeOverflow)?;
        self.layout(layout.pad_to_align())?;
        Ok(Arc::new(value))
    }
    pub(crate) fn grow_vec<T>(
        self,
        values: &mut Vec<T>,
        additional: usize,
    ) -> Result<(), AllocationError> {
        let needed = values
            .len()
            .checked_add(additional)
            .ok_or(AllocationError::SizeOverflow)?;
        if needed <= values.capacity() {
            return Ok(());
        }
        let capacity = values
            .capacity()
            .checked_mul(2)
            .ok_or(AllocationError::SizeOverflow)?
            .max(needed);
        self.layout(Layout::array::<T>(capacity).map_err(|_| AllocationError::SizeOverflow)?)?;
        values
            .try_reserve_exact(capacity - values.len())
            .map_err(|_| AllocationError::HostAllocation)
    }
    pub(crate) fn push<T>(self, values: &mut Vec<T>, value: T) -> Result<(), AllocationError> {
        self.grow_vec(values, 1)?;
        values.push(value);
        Ok(())
    }
    pub(crate) fn push_back<T>(
        self,
        values: &mut VecDeque<T>,
        value: T,
    ) -> Result<(), AllocationError> {
        let needed = values
            .len()
            .checked_add(1)
            .ok_or(AllocationError::SizeOverflow)?;
        if needed > values.capacity() {
            let capacity = values
                .capacity()
                .checked_mul(2)
                .ok_or(AllocationError::SizeOverflow)?
                .max(needed);
            self.layout(Layout::array::<T>(capacity).map_err(|_| AllocationError::SizeOverflow)?)?;
            values
                .try_reserve_exact(capacity - values.len())
                .map_err(|_| AllocationError::HostAllocation)?;
        }
        values.push_back(value);
        Ok(())
    }
    pub(crate) fn reserve_map<K: Eq + Hash, V>(
        self,
        map: &mut hashbrown::HashMap<K, V>,
        additional: usize,
    ) -> Result<(), AllocationError> {
        if let Some(layout) = map
            .try_reserve_layout(additional)
            .map_err(|_| AllocationError::SizeOverflow)?
        {
            self.layout(layout)?;
            map.try_reserve(additional)
                .map_err(|_| AllocationError::HostAllocation)?;
        }
        Ok(())
    }
    pub(crate) fn insert<K: Eq + Hash, V>(
        self,
        map: &mut hashbrown::HashMap<K, V>,
        key: K,
        value: V,
    ) -> Result<Option<V>, AllocationError> {
        if !map.contains_key(&key) {
            self.reserve_map(map, 1)?;
        }
        Ok(map.insert(key, value))
    }
    pub(crate) fn insert_set<K: Eq + Hash>(
        self,
        set: &mut hashbrown::HashSet<K>,
        value: K,
    ) -> Result<bool, AllocationError> {
        if !set.contains(&value) {
            if let Some(layout) = set
                .try_reserve_layout(1)
                .map_err(|_| AllocationError::SizeOverflow)?
            {
                self.layout(layout)?;
                set.try_reserve(1)
                    .map_err(|_| AllocationError::HostAllocation)?;
            }
        }
        Ok(set.insert(value))
    }
}

#[cfg(test)]
mod tests;

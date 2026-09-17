//! Allocation-free prospective layout through the ordinary reservation choice.
#![forbid(unsafe_code)]
use super::*;

impl RawTableInner {
    // Called only after additional exceeds growth_left, by both the ordinary
    // reserve worker and its cold allocation query. Inner None means rehash in
    // place; outer None preserves the worker's capacity-overflow distinction.
    pub(super) fn reserve_rehash_capacity(&self, additional: usize) -> Option<Option<usize>> {
        let new_items = self.items.checked_add(additional)?;
        let full_capacity = bucket_mask_to_capacity(self.bucket_mask);
        if new_items <= full_capacity / 2 {
            Some(None)
        } else {
            Some(Some(usize::max(new_items, full_capacity + 1)))
        }
    }
}
impl<T, A: Allocator> RawTable<T, A> {
    pub(crate) fn try_reserve_layout(&self, additional: usize) -> Result<Option<Layout>, TryReserveError> {
        if additional <= self.table.growth_left {
            return Ok(None);
        }
        let Some(capacity) = self.table.reserve_rehash_capacity(additional)
            .ok_or(TryReserveError::CapacityOverflow)? else {
                return Ok(None);
            };
        let buckets = capacity_to_buckets(capacity, Self::TABLE_LAYOUT)
            .ok_or(TryReserveError::CapacityOverflow)?;
        let (layout, _) = Self::TABLE_LAYOUT.calculate_layout_for(buckets)
            .ok_or(TryReserveError::CapacityOverflow)?;
        Ok(Some(layout))
    }
}

#![forbid(unsafe_code)]
use super::HashTable;
use crate::TryReserveError;
use core::mem::{align_of, size_of};

#[test]
fn prospective_table_layout_matches_requested_growth_and_preserves_entries() {
    #[repr(align(64))]
    struct Aligned(u32);
    fn check<T>(make: impl Fn(u32) -> T, key: impl Fn(&T) -> u32) {
        let mut table = HashTable::new();
        assert!(table.try_reserve_layout(0).unwrap().is_none());
        for id in 0..129u32 {
            let previous = table.allocation_size();
            let quote = table.try_reserve_layout(1).unwrap();
            assert_eq!(table.len(), id as usize);
            assert_eq!(table.allocation_size(), previous, "cold query does not allocate");
            table.try_reserve(1, |value| u64::from(key(value))).unwrap();
            match quote {
                Some(layout) => {
                    assert_eq!(table.allocation_size(), layout.size());
                    assert!(layout.align() >= align_of::<T>());
                    assert!(layout.size() >= (id as usize + 1) * size_of::<T>());
                }
                None => assert_eq!(table.allocation_size(), previous),
            }
            table.insert_unique(u64::from(id), make(id), |value| u64::from(key(value)));
            for prior in 0..=id {
                assert!(table.find(u64::from(prior), |value| key(value) == prior).is_some());
            }
        }
        let before = table.allocation_size();
        assert!(matches!(table.try_reserve_layout(usize::MAX), Err(TryReserveError::CapacityOverflow)));
        assert_eq!(table.allocation_size(), before);
        assert_eq!(table.len(), 129);
    }
    check(|id| id, |value| *value);
    check(Aligned, |value| value.0);
    // ZSTs still own control bytes and use the same requested-layout producer.
    let table = HashTable::<()>::new();
    let quote = table.try_reserve_layout(1).unwrap().unwrap();
    let allocated = HashTable::<()>::with_capacity(1);
    assert_eq!(allocated.allocation_size(), quote.size());
}

#[test]
fn prospective_table_layout_tracks_deleted_slots_and_rehashes_without_probe_allocation() {
    let mut table = HashTable::with_capacity(64);
    let capacity = table.capacity();
    for id in 0..capacity {
        table.insert_unique(id as u64, id, |id| *id as u64);
    }
    for id in 0..capacity-1 {
        table.find_entry(id as u64, |value| *value == id).unwrap().remove();
    }
    let before = table.allocation_size();
    assert!(table.try_reserve_layout(1).unwrap().is_none());
    table.try_reserve(1, |id| *id as u64).unwrap();
    assert_eq!(table.allocation_size(), before);
    assert!(table.find((capacity-1) as u64, |value| *value == capacity-1).is_some());
}

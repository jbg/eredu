use super::*;
use std::error::Error;

#[derive(Debug, PartialEq, Eq)]
struct State {
    backing: Vec<u32>,
    elements: Vec<(u32, u32)>,
    ids: Vec<u32>,
    current: (u32, u32),
    capacities: (usize, usize, usize, usize),
    backing_pointer: *const u32,
    elements_pointer: *const Element,
}

fn state(table: &VecHashCons) -> State {
    let mut ids = table.table.iter().copied().collect::<Vec<_>>();
    ids.sort_unstable();
    State {
        backing: table.backing.clone(),
        elements: table
            .elements
            .iter()
            .map(|element| (element.backing_start, element.backing_end))
            .collect(),
        ids,
        current: (table.curr_elt.backing_start, table.curr_elt.backing_end),
        capacities: (
            table.backing.capacity(),
            table.elements.capacity(),
            table.table.capacity(),
            table.table.allocation_size(),
        ),
        backing_pointer: table.backing.as_ptr(),
        elements_pointer: table.elements.as_ptr(),
    }
}

#[test]
fn retained_capacity_accounts_for_all_spare_backing_and_table_allocation() {
    let mut table = VecHashCons::new();
    assert_eq!(table.retained_capacity_bytes().unwrap(), 0);
    table.reserve(19);
    let before = table.retained_capacity_bytes().unwrap();
    assert!(before > 0);
    let first = table.insert(&[7, 11, 13]);
    assert_eq!(table.get(first), &[7, 11, 13]);
    assert!(table.backing.capacity() > 3);
    assert!(table.elements.capacity() > table.elements.len());
    assert!(table.table.allocation_size() > 0);
    assert_eq!(
        table.retained_capacity_bytes().unwrap(),
        table.backing.capacity() * std::mem::size_of::<u32>()
            + table.elements.capacity() * std::mem::size_of::<Element>()
            + table.table.allocation_size()
    );
    let copy = table.clone();
    assert_eq!(copy.get(first), &[7, 11, 13]);
    assert_eq!(
        copy.retained_capacity_bytes().unwrap(),
        copy.backing.capacity() * std::mem::size_of::<u32>()
            + copy.elements.capacity() * std::mem::size_of::<Element>()
            + copy.table.allocation_size()
    );
    assert_eq!(state(&table).ids, state(&copy).ids);
}

#[test]
fn lookup_is_read_only_even_during_legacy_incremental_insertion() {
    let mut table = VecHashCons::new();
    let first = table.insert(&[3, 5, 7]);
    let empty = table.insert(&[]);
    table.start_insert();
    table.push_u32(11);
    table.push_slice(&[13, 17]);
    let before = state(&table);
    assert_eq!(table.lookup(&[3, 5, 7]), Some(first));
    assert_eq!(table.lookup(&[]), Some(empty));
    assert_eq!(table.lookup(&[11, 13, 17]), None);
    assert_eq!(table.lookup(&[3, 5]), None);
    assert_eq!(state(&table), before);
    let next = table.finish_insert();
    assert_eq!(next, 2);
    assert_eq!(table.get(next), &[11, 13, 17]);
    assert_eq!(table.lookup(&[11, 13, 17]), Some(next));
    assert_eq!(table.insert(&[3, 5, 7]), first);
}

#[test]
fn exact_words_and_entries_accept_while_one_short_rejects_unchanged() {
    let mut exact = PreparedVecHashCons::try_new(6, 2).unwrap();
    let first = exact.try_insert(&[2, 3, 5]).unwrap();
    let second = exact.try_insert(&[7, 11, 13]).unwrap();
    assert_eq!((first, second), (0, 1));
    assert_eq!(exact.get(second), &[7, 11, 13]);
    assert!(exact.is_valid(first));
    assert!(!exact.is_valid(2));
    assert_eq!(exact.max_words(), 6);
    assert_eq!(exact.max_entries(), 2);

    let mut short_words = PreparedVecHashCons::try_new(5, 2).unwrap();
    short_words.try_insert(&[2, 3, 5]).unwrap();
    let before = state(&short_words.inner);
    assert!(matches!(
        short_words.try_insert(&[7, 11, 13]),
        Err(HashConsCapacityError::WordsExceeded {
            required_words: 6,
            capacity_words: 5
        })
    ));
    assert_eq!(state(&short_words.inner), before);
    assert_eq!(short_words.lookup(&[7, 11, 13]), None);
    assert_eq!(short_words.get(0), &[2, 3, 5]);

    let mut short_entries = PreparedVecHashCons::try_new(6, 1).unwrap();
    short_entries.try_insert(&[2, 3, 5]).unwrap();
    let before = state(&short_entries.inner);
    assert!(matches!(
        short_entries.try_insert(&[7, 11, 13]),
        Err(HashConsCapacityError::EntriesExceeded {
            required_entries: 2,
            capacity_entries: 1
        })
    ));
    assert_eq!(state(&short_entries.inner), before);
    assert_eq!(short_entries.lookup(&[7, 11, 13]), None);
}

#[test]
fn duplicate_lookup_succeeds_when_every_prepared_allowance_is_full() {
    let mut table = PreparedVecHashCons::try_new(3, 2).unwrap();
    let empty = table.try_insert(&[]).unwrap();
    let value = table.try_insert(&[19, 23, 29]).unwrap();
    let before = state(&table.inner);
    assert_eq!(table.len(), 2);
    assert_eq!(table.try_insert(&[]).unwrap(), empty);
    assert_eq!(table.try_insert(&[19, 23, 29]).unwrap(), value);
    assert_eq!(table.lookup(&[19, 23, 29]), Some(value));
    assert_eq!(state(&table.inner), before);
    assert!(matches!(
        table.try_insert(&[31]),
        Err(HashConsCapacityError::WordsExceeded { .. })
    ));
    assert_eq!(state(&table.inner), before);
}

#[test]
fn repeated_insertion_preserves_prepared_storage_and_matches_legacy_interning() {
    let mut fixed = PreparedVecHashCons::try_new(512, 64).unwrap();
    let mut legacy = VecHashCons::new();
    let backing_pointer = fixed.inner.backing.as_ptr();
    let elements_pointer = fixed.inner.elements.as_ptr();
    let capacity = fixed.retained_capacity_bytes().unwrap();
    let table_capacity = fixed.inner.table.capacity();
    let table_bytes = fixed.inner.table.allocation_size();
    assert!(fixed.is_empty());
    for n in 0..64u32 {
        let data = [n + 1, n * 7, 0x7654_3210 ^ n, u32::MAX - n];
        let actual = fixed.try_insert(&data).unwrap();
        assert_eq!(actual, legacy.insert(&data));
        assert_eq!(fixed.lookup(&data), Some(actual));
        assert_eq!(fixed.get(actual), legacy.get(actual));
        assert_eq!(fixed.try_insert(&data).unwrap(), actual);
        assert_eq!(fixed.inner.backing.as_ptr(), backing_pointer);
        assert_eq!(fixed.inner.elements.as_ptr(), elements_pointer);
        assert_eq!(fixed.inner.table.capacity(), table_capacity);
        assert_eq!(fixed.inner.table.allocation_size(), table_bytes);
        assert_eq!(fixed.retained_capacity_bytes().unwrap(), capacity);
    }
    assert_eq!(fixed.len(), 64);
}

#[test]
fn zero_allowances_and_representation_overflow_reject_before_insertion() {
    let mut zero = PreparedVecHashCons::try_new(0, 0).unwrap();
    let before = state(&zero.inner);
    assert!(matches!(
        zero.try_insert(&[]),
        Err(HashConsCapacityError::EntriesExceeded {
            required_entries: 1,
            capacity_entries: 0
        })
    ));
    assert_eq!(state(&zero.inner), before);
    let mut empty_only = PreparedVecHashCons::try_new(0, 1).unwrap();
    assert_eq!(empty_only.try_insert(&[]).unwrap(), 0);
    assert_eq!(empty_only.try_insert(&[]).unwrap(), 0);
    assert!(matches!(
        empty_only.try_insert(&[1]),
        Err(HashConsCapacityError::WordsExceeded { .. })
    ));
    assert!(matches!(
        PreparedVecHashCons::try_new(usize::MAX, 0),
        Err(HashConsCapacityError::CapacityOverflow)
    ));
    if usize::BITS > 32 {
        assert!(matches!(
            PreparedVecHashCons::try_new(u32::MAX as usize, 0),
            Err(HashConsCapacityError::IndexOverflow)
        ));
        let too_many = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
        assert!(matches!(
            PreparedVecHashCons::try_new(0, too_many),
            Err(HashConsCapacityError::IndexOverflow)
        ));
    }
}

#[test]
fn invalid_prepared_scratch_and_index_fail_without_changing_any_storage() {
    let mut table = PreparedVecHashCons::try_new(4, 2).unwrap();
    table.try_insert(&[31, 37]).unwrap();
    // Private-state fault injection exercises the defensive guards without
    // exposing an unchecked mutation or constructing enormous allocations.
    table.inner.curr_elt.backing_end = 3;
    let before = state(&table.inner);
    assert!(matches!(
        table.try_insert(&[41]),
        Err(HashConsCapacityError::InvalidPreparedStorage)
    ));
    assert_eq!(state(&table.inner), before);
    table.inner.curr_elt.backing_end = 0;
    table.inner.curr_elt.backing_start = u32::MAX;
    let before = state(&table.inner);
    assert!(matches!(
        table.try_insert(&[41]),
        Err(HashConsCapacityError::IndexOverflow | HashConsCapacityError::CapacityOverflow)
    ));
    assert_eq!(state(&table.inner), before);
}

#[test]
fn reservation_errors_preserve_original_vector_and_table_sources() {
    let vector = Vec::<u32>::new().try_reserve_exact(usize::MAX).unwrap_err();
    let error = HashConsCapacityError::VectorAllocation(vector);
    assert!(error.source().unwrap().is::<TryReserveError>());
    let table = HashTable::<u32>::new()
        .try_reserve(usize::MAX, |_| 0)
        .unwrap_err();
    let error = HashConsCapacityError::TableAllocation(table);
    assert!(error.source().unwrap().is::<hashbrown::TryReserveError>());
    assert!(matches!(
        allocation_bytes::<u32>(usize::MAX),
        Err(HashConsCapacityError::CapacityOverflow)
    ));
}

#[test]
fn legacy_geometry_validation_rejects_overflow_and_active_scratch_without_mutation() {
    let mut table = VecHashCons::new();
    let before = state(&table);
    table.validate_insert_geometry(3).unwrap();
    assert_eq!(state(&table), before);
    assert!(matches!(
        table.validate_insert_geometry(usize::MAX),
        Err(HashConsCapacityError::IndexOverflow)
    ));
    assert!(matches!(
        table.validate_insert_geometry(u32::MAX as usize),
        Err(HashConsCapacityError::IndexOverflow)
    ));
    assert_eq!(state(&table), before);
    table.insert(&[2, 3, 5]);
    table.start_insert();
    table.push_slice(&[7, 11]);
    let before = state(&table);
    assert!(matches!(
        table.validate_insert_geometry(1),
        Err(HashConsCapacityError::InsertionInProgress)
    ));
    assert_eq!(state(&table), before);
    table.finish_insert();
    table.validate_insert_geometry(3).unwrap();
    assert_eq!(table.lookup(&[7, 11]), Some(1));
}

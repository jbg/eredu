use super::*;
use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};
#[derive(Debug)]
struct EntryDenied;
impl fmt::Display for EntryDenied {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str("entry growth denied") }
}
impl std::error::Error for EntryDenied {}

#[test]
fn paid_entry_growth_preserves_staged_words_ids_and_duplicate_zero_spend() {
    let spent = Arc::new(AtomicUsize::new(0));
    let denied = Arc::new(AtomicBool::new(false));
    let owner = {
        let spent = spent.clone(); let denied = denied.clone();
        ParserAllocationFunding::prepare(move |bytes| {
            if denied.load(Ordering::SeqCst) { Err(EntryDenied) }
            else { spent.fetch_add(bytes, Ordering::SeqCst); Ok(()) }
        }).unwrap()
    };
    let mut table = PreparedVecHashCons::try_new_with_scratch(512, 1, 2).unwrap();
    table.bind_backing_funding(owner).unwrap();
    for id in 0..65u32 {
        let data = [id, id ^ 85];
        let quote = table.inner.table.try_reserve_layout(1).unwrap();
        let prior = table.inner.table.allocation_size();
        let mut write = table.begin_insert(2).unwrap();
        write.push_slice(&data).unwrap();
        let staged = write.owner.inner.backing[write.start..].as_ptr();
        let actual = write.finish().unwrap();
        assert_eq!(actual, id);
        assert_eq!(table.get(id).as_ptr(), staged, "publication keeps actual staged words");
        assert_eq!(table.get(id), &data);
        assert_eq!(table.inner.table.allocation_size(), quote.map_or(prior, |layout| layout.size()));
        for earlier in 0..=id { assert_eq!(table.lookup(&[earlier, earlier ^ 85]), Some(earlier)); }
    }
    let paid = spent.load(Ordering::SeqCst);
    denied.store(true, Ordering::SeqCst);
    assert_eq!(table.try_insert(&[3, 3 ^ 85]).unwrap(), 3);
    let mut duplicate = table.begin_insert(2).unwrap();
    duplicate.push_slice(&[7, 7 ^ 85]).unwrap();
    assert_eq!(duplicate.finish().unwrap(), 7);
    assert_eq!(spent.load(Ordering::SeqCst), paid);
    let length = table.len(); let entries = table.max_entries();
    let mut write = table.begin_insert(2).unwrap();
    write.push_slice(&[999, 1000]).unwrap();
    let error = write.finish().unwrap_err();
    assert_eq!(table.len(), length); assert_eq!(table.max_entries(), entries);
    assert_eq!(table.lookup(&[999, 1000]), None);
    assert!(!table.insertion_active);
    assert!(table.inner.backing[table.inner.curr_elt.backing_start as usize..].iter().all(|word| *word == 0));
    let mut cause: &dyn std::error::Error = &error;
    while cause.downcast_ref::<EntryDenied>().is_none() { cause = cause.source().expect("original paid refusal"); }
    assert_eq!(table.try_insert(&[3, 3 ^ 85]).unwrap(), 3);
    let detached = table.inner.prepared_source_plan().unwrap().compile().unwrap();
    assert!(detached.backing_funding.is_none(), "source copies never inherit spending authority");
    drop((table, detached, spent, denied));
    let mut cause: &dyn std::error::Error = &error;
    while cause.downcast_ref::<EntryDenied>().is_none() { cause = cause.source().expect("escaped original refusal"); }
}

#[test]
fn fixed_entry_allowance_still_refuses_without_paid_owner() {
    let mut table = PreparedVecHashCons::try_new_with_scratch(8, 1, 2).unwrap();
    assert_eq!(table.try_insert(&[1]).unwrap(), 0);
    assert!(matches!(table.try_insert(&[2]), Err(HashConsCapacityError::EntriesExceeded { .. })));
    assert_eq!(table.try_insert(&[1]).unwrap(), 0);
}

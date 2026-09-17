use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Debug, PartialEq, Eq)]
struct Snapshot {
    backing: Vec<u32>,
    elements: Vec<(u32, u32)>,
    ids: Vec<u32>,
    current: (u32, u32),
    active: bool,
    capacities: (usize, usize, usize, usize),
    pointers: (*const u32, *const Element),
}

fn snapshot(owner: &PreparedVecHashCons) -> Snapshot {
    let inner = &owner.inner;
    let mut ids: Vec<_> = inner.table.iter().copied().collect();
    ids.sort_unstable();
    Snapshot {
        backing: inner.backing.clone(),
        elements: inner
            .elements
            .iter()
            .map(|element| (element.backing_start, element.backing_end))
            .collect(),
        ids,
        current: (inner.curr_elt.backing_start, inner.curr_elt.backing_end),
        active: owner.insertion_active,
        capacities: (
            inner.backing.capacity(),
            inner.elements.capacity(),
            inner.table.capacity(),
            inner.table.allocation_size(),
        ),
        pointers: (inner.backing.as_ptr(), inner.elements.as_ptr()),
    }
}

fn assert_clean_tail(owner: &PreparedVecHashCons) {
    assert!(
        owner.inner.backing[owner.inner.curr_elt.backing_start as usize..]
            .iter()
            .all(|word| *word == 0)
    );
    assert!(!owner.insertion_active);
}

fn insert_chunks(owner: &mut PreparedVecHashCons, words: &[u32], split: usize) -> u32 {
    let mut insertion = owner.begin_insert(words.len()).unwrap();
    insertion.push_slice(&words[..split]).unwrap();
    for &word in &words[split..] {
        insertion.push_u32(word).unwrap();
    }
    insertion.finish().unwrap()
}

#[test]
fn explicit_scratch_is_retained_and_exact_or_one_short_is_checked_before_writing() {
    let mut exact = PreparedVecHashCons::try_new_with_scratch(3, 1, 3).unwrap();
    assert_eq!(exact.max_encoded_words(), 3);
    assert_eq!(exact.inner.backing.len(), 4 + 3 + 3);
    assert_eq!(
        exact.retained_capacity_bytes().unwrap(),
        exact.inner.backing.capacity() * std::mem::size_of::<u32>()
            + exact.inner.elements.capacity() * std::mem::size_of::<Element>()
            + exact.inner.table.allocation_size(),
    );
    assert_eq!(insert_chunks(&mut exact, &[3, 5, 7], 1), 0);
    assert_eq!(exact.get(0), &[3, 5, 7]);
    assert_clean_tail(&exact);

    let mut short = PreparedVecHashCons::try_new_with_scratch(3, 1, 2).unwrap();
    let before = snapshot(&short);
    assert!(matches!(
        short.begin_insert(3),
        Err(HashConsCapacityError::ScratchExceeded {
            required_words: 3,
            capacity_words: 2,
        })
    ));
    assert_eq!(snapshot(&short), before);

    let mut direct = PreparedVecHashCons::try_new(3, 1).unwrap();
    assert_eq!(direct.max_encoded_words(), 0);
    assert!(matches!(
        direct.begin_insert(1),
        Err(HashConsCapacityError::ScratchExceeded {
            capacity_words: 0,
            ..
        })
    ));
    assert_eq!(direct.try_insert(&[3, 5, 7]).unwrap(), 0);
    assert_eq!(direct.get(0), exact.get(0));
}

#[test]
fn publication_commits_the_staged_backing_without_copy_or_early_visibility() {
    let mut owner = PreparedVecHashCons::try_new_with_scratch(6, 2, 3).unwrap();
    let first = owner.try_insert(&[11, 13, 17]).unwrap();
    let before = snapshot(&owner);
    let bytes = owner.retained_capacity_bytes().unwrap();
    let mut insertion = owner.begin_insert(3).unwrap();
    insertion.push_u32(19).unwrap();
    insertion.push_slice(&[23, 29]).unwrap();
    let staged_pointer = insertion.owner.inner.backing[insertion.start..].as_ptr();
    assert_eq!(insertion.owner.len(), 1);
    assert_eq!(insertion.owner.lookup(&[19, 23, 29]), None);
    assert_eq!(insertion.owner.get(first), &[11, 13, 17]);
    assert_eq!(
        insertion.owner.inner.curr_elt.backing_start,
        before.current.0
    );
    let second = insertion.finish().unwrap();
    assert_eq!(second, 1);
    assert_eq!(owner.get(second), &[19, 23, 29]);
    assert_eq!(owner.get(second).as_ptr(), staged_pointer);
    assert_eq!(owner.lookup(&[19, 23, 29]), Some(second));
    assert_eq!(snapshot(&owner).capacities, before.capacities);
    assert_eq!(snapshot(&owner).pointers, before.pointers);
    assert_eq!(owner.retained_capacity_bytes().unwrap(), bytes);
    assert_clean_tail(&owner);
}

#[test]
fn duplicate_at_full_committed_capacity_clears_staging_and_new_value_rejects_unchanged() {
    let mut owner = PreparedVecHashCons::try_new_with_scratch(3, 2, 3).unwrap();
    let empty = owner.begin_insert(0).unwrap().finish().unwrap();
    let first = insert_chunks(&mut owner, &[31, 37, 41], 2);
    let before = snapshot(&owner);
    assert_eq!(owner.begin_insert(0).unwrap().finish().unwrap(), empty);
    assert_eq!(insert_chunks(&mut owner, &[31, 37, 41], 0), first);
    assert_eq!(snapshot(&owner), before);

    let mut insertion = owner.begin_insert(3).unwrap();
    insertion.push_slice(&[43, 47, 53]).unwrap();
    assert!(matches!(
        insertion.finish(),
        Err(HashConsCapacityError::WordsExceeded {
            required_words: 6,
            capacity_words: 3,
        })
    ));
    assert_eq!(snapshot(&owner), before);
    assert_eq!(owner.lookup(&[43, 47, 53]), None);
    assert_clean_tail(&owner);
}

#[test]
fn entry_limit_and_direct_insertion_cannot_consume_reserved_scratch_as_payload() {
    let mut owner = PreparedVecHashCons::try_new_with_scratch(8, 1, 4).unwrap();
    insert_chunks(&mut owner, &[59, 61], 1);
    let before = snapshot(&owner);
    let mut insertion = owner.begin_insert(2).unwrap();
    insertion.push_slice(&[67, 71]).unwrap();
    assert!(matches!(
        insertion.finish(),
        Err(HashConsCapacityError::EntriesExceeded {
            required_entries: 2,
            capacity_entries: 1,
        })
    ));
    assert_eq!(snapshot(&owner), before);
    assert_clean_tail(&owner);

    let mut small = PreparedVecHashCons::try_new_with_scratch(2, 4, 8).unwrap();
    small.try_insert(&[73, 79]).unwrap();
    let before = snapshot(&small);
    assert!(matches!(
        small.try_insert(&[83]),
        Err(HashConsCapacityError::WordsExceeded {
            required_words: 3,
            capacity_words: 2,
        })
    ));
    assert_eq!(snapshot(&small), before);
}

#[test]
fn incomplete_and_overlong_encodings_clear_scratch_and_ignored_errors_remain_sticky() {
    let mut owner = PreparedVecHashCons::try_new_with_scratch(8, 3, 4).unwrap();
    owner.try_insert(&[89, 97]).unwrap();
    let before = snapshot(&owner);
    let mut incomplete = owner.begin_insert(3).unwrap();
    incomplete.push_slice(&[101, 103]).unwrap();
    assert!(matches!(
        incomplete.finish(),
        Err(HashConsCapacityError::EncodedLengthMismatch {
            expected_words: 3,
            actual_words: 2,
        })
    ));
    assert_eq!(snapshot(&owner), before);

    let mut overlong = owner.begin_insert(3).unwrap();
    overlong.push_slice(&[107, 109]).unwrap();
    assert!(matches!(
        overlong.push_slice(&[113, 127]),
        Err(HashConsCapacityError::EncodedLengthMismatch {
            expected_words: 3,
            actual_words: 4,
        })
    ));
    // Cleanup occurs on rejection, even while the failed guard remains alive.
    assert!(overlong.owner.inner.backing[overlong.start..]
        .iter()
        .all(|v| *v == 0));
    for attempt in [overlong.push_u32(131), overlong.push_slice(&[])] {
        assert!(matches!(
            attempt,
            Err(HashConsCapacityError::EncodedLengthMismatch {
                expected_words: 3,
                actual_words: 4,
            })
        ));
    }
    assert!(matches!(
        overlong.finish(),
        Err(HashConsCapacityError::EncodedLengthMismatch {
            expected_words: 3,
            actual_words: 4,
        })
    ));
    assert_eq!(snapshot(&owner), before);

    // Ignoring an extra push after a fully written value must not commit that
    // otherwise valid prefix, even if it is an existing interned value.
    let mut extra = owner.begin_insert(2).unwrap();
    extra.push_slice(&[89, 97]).unwrap();
    assert!(extra.push_u32(137).is_err());
    assert!(matches!(
        extra.finish(),
        Err(HashConsCapacityError::EncodedLengthMismatch {
            expected_words: 2,
            actual_words: 3,
        })
    ));
    assert_eq!(snapshot(&owner), before);
    assert_eq!(insert_chunks(&mut owner, &[139, 149, 151], 2), 1);
    assert_clean_tail(&owner);
}

#[test]
fn abandoned_and_unwound_writers_restore_all_scratch_and_allow_subsequent_insertion() {
    let mut owner = PreparedVecHashCons::try_new_with_scratch(8, 3, 4).unwrap();
    owner.try_insert(&[157, 163]).unwrap();
    let before = snapshot(&owner);
    {
        let mut abandoned = owner.begin_insert(4).unwrap();
        abandoned.push_slice(&[167, 173, 179]).unwrap();
    }
    assert_eq!(snapshot(&owner), before);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let mut unwound = owner.begin_insert(4).unwrap();
        unwound.push_u32(181).unwrap();
        unwound.push_slice(&[191, 193]).unwrap();
        panic!("serializer stopped before finishing its encoding");
    }));
    assert!(panic.is_err());
    assert_eq!(snapshot(&owner), before);
    assert_eq!(insert_chunks(&mut owner, &[197, 199, 211, 223], 2), 1);
    assert_clean_tail(&owner);
}

#[test]
fn forgotten_writer_fences_every_mutating_entry_point_without_publishing_scratch() {
    let mut owner = PreparedVecHashCons::try_new_with_scratch(8, 3, 4).unwrap();
    let first = owner.try_insert(&[227, 229]).unwrap();
    let mut forgotten = owner.begin_insert(4).unwrap();
    forgotten.push_slice(&[233, 239]).unwrap();
    std::mem::forget(forgotten);
    let stranded = snapshot(&owner);
    assert!(owner.insertion_active);
    assert_eq!(owner.len(), 1);
    assert_eq!(owner.get(first), &[227, 229]);
    assert_eq!(owner.lookup(&[227, 229]), Some(first));
    assert_eq!(owner.lookup(&[233, 239]), None);
    for value in [&[227, 229][..], &[241][..], &[][..]] {
        assert!(matches!(
            owner.try_insert(value),
            Err(HashConsCapacityError::InsertionInProgress)
        ));
    }
    assert!(matches!(
        owner.begin_insert(0),
        Err(HashConsCapacityError::InsertionInProgress)
    ));
    assert!(matches!(
        owner.begin_insert(4),
        Err(HashConsCapacityError::InsertionInProgress)
    ));
    assert_eq!(snapshot(&owner), stranded);
    // The owner still owns every allocation; no recovery or early refund is
    // claimed for a deliberately forgotten writer.
    drop(owner);
}

#[test]
fn empty_encodings_need_no_scratch_but_still_obey_entry_and_exact_length_limits() {
    let mut owner = PreparedVecHashCons::try_new_with_scratch(0, 1, 0).unwrap();
    let mut empty = owner.begin_insert(0).unwrap();
    empty.push_slice(&[]).unwrap();
    assert_eq!(empty.finish().unwrap(), 0);
    let before = snapshot(&owner);
    assert_eq!(owner.begin_insert(0).unwrap().finish().unwrap(), 0);
    let mut overlong = owner.begin_insert(0).unwrap();
    assert!(matches!(
        overlong.push_u32(251),
        Err(HashConsCapacityError::EncodedLengthMismatch {
            expected_words: 0,
            actual_words: 1,
        })
    ));
    assert!(overlong.finish().is_err());
    assert_eq!(snapshot(&owner), before);
    assert_clean_tail(&owner);

    let mut no_entries = PreparedVecHashCons::try_new_with_scratch(0, 0, 0).unwrap();
    let before = snapshot(&no_entries);
    assert!(matches!(
        no_entries.begin_insert(0).unwrap().finish(),
        Err(HashConsCapacityError::EntriesExceeded {
            required_entries: 1,
            capacity_entries: 0,
        })
    ));
    assert_eq!(snapshot(&no_entries), before);
}

#[test]
fn overflowing_and_invalid_staging_geometry_rejects_before_state_changes() {
    assert!(matches!(
        PreparedVecHashCons::try_new_with_scratch(0, 0, usize::MAX),
        Err(HashConsCapacityError::CapacityOverflow)
    ));
    if usize::BITS > 32 {
        assert!(matches!(
            PreparedVecHashCons::try_new_with_scratch(0, 0, u32::MAX as usize),
            Err(HashConsCapacityError::IndexOverflow)
        ));
    }
    let mut owner = PreparedVecHashCons::try_new_with_scratch(4, 2, 3).unwrap();
    owner.try_insert(&[257, 263]).unwrap();
    let before = snapshot(&owner);
    assert!(matches!(
        owner.begin_insert(usize::MAX),
        Err(HashConsCapacityError::CapacityOverflow)
    ));
    assert!(matches!(
        owner.begin_insert(u32::MAX as usize),
        Err(HashConsCapacityError::CapacityOverflow | HashConsCapacityError::IndexOverflow)
    ));
    assert_eq!(snapshot(&owner), before);

    // Inject inconsistent private backing without a large allocation. Cold
    // validation must reject before creating an active writer or touching it.
    owner.inner.backing.truncate(7);
    let before = snapshot(&owner);
    assert!(matches!(
        owner.begin_insert(3),
        Err(HashConsCapacityError::InvalidPreparedStorage)
    ));
    assert_eq!(snapshot(&owner), before);
}

#[test]
fn repeated_segmented_encodings_match_legacy_ids_and_keep_all_capacities_fixed() {
    let mut owner = PreparedVecHashCons::try_new_with_scratch(256, 64, 4).unwrap();
    let mut legacy = VecHashCons::new();
    let initial = snapshot(&owner);
    let bytes = owner.retained_capacity_bytes().unwrap();
    for n in 0..64u32 {
        let words = [n + 1, n * 17, 0x7654_3210 ^ n, u32::MAX - n];
        let actual = insert_chunks(&mut owner, &words, n as usize % 5);
        assert_eq!(actual, legacy.insert(&words));
        assert_eq!(owner.get(actual), legacy.get(actual));
        assert_eq!(owner.lookup(&words), Some(actual));
        let before_duplicate = snapshot(&owner);
        assert_eq!(
            insert_chunks(&mut owner, &words, (n as usize + 2) % 5),
            actual
        );
        assert_eq!(snapshot(&owner), before_duplicate);
        assert_eq!(snapshot(&owner).capacities, initial.capacities);
        assert_eq!(snapshot(&owner).pointers, initial.pointers);
        assert_eq!(owner.retained_capacity_bytes().unwrap(), bytes);
        assert_clean_tail(&owner);
    }
    assert_eq!(owner.len(), 64);
    assert_eq!(owner.inner.curr_elt.backing_start, 260);
}

#[test]
fn funded_reached_backing_growth_keeps_ids_scratch_and_exact_refusal_owner() {
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    };
    #[derive(Debug)]
    struct Denied;
    impl std::fmt::Display for Denied {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("denied backing")
        }
    }
    impl std::error::Error for Denied {}
    struct Account {
        deny: AtomicBool,
        spent: AtomicUsize,
        calls: AtomicUsize,
        retired: Arc<AtomicBool>,
    }
    impl Drop for Account {
        fn drop(&mut self) {
            self.retired.store(true, Ordering::SeqCst);
        }
    }
    fn owner(account: &Arc<Account>) -> PreparedHashConsFunding {
        let account = Arc::clone(account);
        PreparedHashConsFunding::prepare(move |bytes| {
            account.calls.fetch_add(1, Ordering::SeqCst);
            if account.deny.load(Ordering::SeqCst) {
                Err(Denied)
            } else {
                account.spent.fetch_add(bytes, Ordering::SeqCst);
                Ok(())
            }
        })
        .unwrap()
    }
    let retired = Arc::new(AtomicBool::new(false));
    let account = Arc::new(Account {
        deny: AtomicBool::new(false),
        spent: AtomicUsize::new(0),
        calls: AtomicUsize::new(0),
        retired: Arc::clone(&retired),
    });
    let mut source = VecHashCons::new();
    let first = source.insert(&[7]);
    let mut table = source.prepared_source_plan().unwrap().compile().unwrap();
    let previous_words = table.max_words();
    let previous_scratch = table.max_encoded_words();
    table.bind_backing_funding(owner(&account)).unwrap();
    let encoding = vec![19; previous_words + 3];
    let before = table.retained_capacity_bytes().unwrap();
    let id = {
        let mut write = table.begin_insert(encoding.len()).unwrap();
        write.push_slice(&encoding).unwrap();
        write.finish().unwrap()
    };
    assert_eq!(id, first + 1);
    assert_eq!(table.get(first), &[7]);
    assert_eq!(table.get(id), encoding.as_slice());
    assert!(table.max_encoded_words() > previous_scratch);
    assert_eq!(table.max_words(), 1 + encoding.len());
    assert!(table.retained_capacity_bytes().unwrap() > before);
    let paid = account.spent.load(Ordering::SeqCst);
    let calls = account.calls.load(Ordering::SeqCst);
    let duplicate = {
        let mut write = table.begin_insert(encoding.len()).unwrap();
        write.push_slice(&encoding).unwrap();
        write.finish().unwrap()
    };
    assert_eq!(duplicate, id);
    assert_eq!(account.spent.load(Ordering::SeqCst), paid);
    assert_eq!(account.calls.load(Ordering::SeqCst), calls);
    let rejected = vec![23; encoding.len()];
    let mut write = table.begin_insert(rejected.len()).unwrap();
    write.push_slice(&rejected).unwrap();
    account.deny.store(true, Ordering::SeqCst);
    let error = write.finish().unwrap_err();
    assert!(matches!(error, HashConsCapacityError::Funding(_)));
    assert_eq!(table.lookup(&rejected), None);
    assert_eq!(table.get(id), encoding.as_slice());
    assert!(!table.insertion_active);
    let mut cause: &dyn std::error::Error = &error;
    let mut exact = false;
    loop {
        exact |= cause.downcast_ref::<Denied>().is_some();
        match cause.source() {
            Some(next) => cause = next,
            None => break,
        }
    }
    assert!(exact);
    drop((table, source, account));
    assert!(!retired.load(Ordering::SeqCst));
    drop(error);
    assert!(retired.load(Ordering::SeqCst));
}

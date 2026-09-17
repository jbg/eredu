use super::*;
use std::{
    cell::Cell,
    collections::BTreeMap,
    panic::{catch_unwind, AssertUnwindSafe},
    rc::Rc,
    sync::{Arc, Mutex},
};

fn check<K: Ord, V, C>(node: &Link<K, V, C>) -> usize {
    let Some(node) = node else { return 0 };
    let l = check(&node.node.left);
    let r = check(&node.node.right);
    assert!(l.abs_diff(r) <= 1);
    assert_eq!(node.node.height, l.max(r) + 1);
    node.node.height
}

#[test]
fn randomized_prepared_index_matches_ordered_map_through_pruning_and_duplicates() {
    let mut index = PreparedIndex::default();
    let mut expected = BTreeMap::new();
    let mut state = 19_u64;
    for step in 0..4096 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let key = (state >> 32) % 257;
        if step % 11 == 0 {
            let modulus = key % 7 + 2;
            let retired = index.extract_if(|key, _| key % modulus == 0);
            expected.retain(|key, _| key % modulus != 0);
            drop(retired);
        } else {
            let mut candidate = Some(PreparedIndexNode::new(key, step, ()));
            let address = std::ptr::from_ref(candidate.as_ref().unwrap().value());
            match expected.entry(key) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    assert!(index.insert(&mut candidate));
                    entry.insert(step);
                    assert_eq!(
                        std::ptr::from_ref(index.get_by(|k| key.cmp(k)).unwrap()),
                        address
                    );
                }
                std::collections::btree_map::Entry::Occupied(entry) => {
                    assert!(!index.insert(&mut candidate));
                    let refused = candidate.take().unwrap();
                    assert_eq!(std::ptr::from_ref(refused.value()), address);
                    assert_eq!(index.get_by(|k| key.cmp(k)), Some(entry.get()));
                }
            }
        }
        assert_eq!(index.len(), expected.len());
        // Sorted borrowed traversal must survive rotations, duplicate refusals
        // and pruning; compare the complete sequence and the original addresses.
        let mut actual = index.iter();
        for (key, value) in &expected {
            let (found_key, found_value) = actual.next().unwrap();
            assert_eq!((found_key, found_value), (key, value));
            assert!(std::ptr::eq(
                found_value,
                index.get_by(|k| key.cmp(k)).unwrap()
            ));
        }
        assert!(actual.next().is_none());
        assert!(actual.next().is_none());
        check(&index.root);
        for key in 0..257 {
            assert_eq!(index.get_by(|k| key.cmp(k)), expected.get(&key));
        }
    }
}

#[test]
fn adversarial_insertions_keep_logarithmic_lookup_and_stable_value_addresses() {
    for descending in [false, true] {
        let mut index = PreparedIndex::default();
        let mut addresses = Vec::new();
        for ordinal in 0..4096_usize {
            let key = if descending { 4095 - ordinal } else { ordinal };
            let mut candidate = Some(PreparedIndexNode::new(key, key * 3, ()));
            addresses.push((key, std::ptr::from_ref(candidate.as_ref().unwrap().value())));
            assert!(index.insert(&mut candidate));
        }
        let depth = check(&index.root);
        assert!(depth <= 2 * (4096_usize.ilog2() as usize + 1));
        for &(key, address) in &addresses {
            let mut comparisons = 0;
            let found = index
                .get_by(|k| {
                    comparisons += 1;
                    key.cmp(k)
                })
                .unwrap();
            assert_eq!(std::ptr::from_ref(found), address);
            assert_eq!(*found, key * 3);
            assert!(comparisons <= depth);
        }
        drop(index.extract_if(|key, _| key % 3 == 0));
        check(&index.root);
        for &(key, address) in &addresses {
            let found = index.get_by(|k| key.cmp(k));
            if key % 3 == 0 {
                assert!(found.is_none())
            } else {
                assert_eq!(std::ptr::from_ref(found.unwrap()), address)
            }
        }
    }
}

#[test]
fn observation_unwind_keeps_tree_and_detached_retirement_runs_outside_lock() {
    #[derive(Debug)]
    struct Value {
        lock: Arc<Mutex<()>>,
        drops: Rc<Cell<usize>>,
        value: usize,
    }
    impl Drop for Value {
        fn drop(&mut self) {
            assert!(self.lock.try_lock().is_ok());
            self.drops.set(self.drops.get() + 1);
        }
    }
    let lock = Arc::new(Mutex::new(()));
    let drops = Rc::new(Cell::new(0));
    let mut index = PreparedIndex::default();
    for key in 0..33 {
        let mut candidate = Some(PreparedIndexNode::new(
            key,
            Value {
                lock: lock.clone(),
                drops: drops.clone(),
                value: key,
            },
            (),
        ));
        assert!(index.insert(&mut candidate));
    }
    let held = lock.lock().unwrap();
    let failed = catch_unwind(AssertUnwindSafe(|| {
        index.extract_if(|key, _| {
            if *key == 17 {
                panic!("observation failed")
            }
            key % 2 == 0
        })
    }));
    assert!(failed.is_err());
    assert_eq!(index.len(), 33);
    assert_eq!(drops.get(), 0);
    for key in 0..33 {
        assert_eq!(index.get_by(|k| key.cmp(k)).unwrap().value, key);
    }
    let retired = index.extract_if(|key, _| key % 2 == 0);
    assert_eq!(index.len(), 16);
    assert_eq!(drops.get(), 0);
    drop(held);
    drop(retired);
    assert_eq!(drops.get(), 17);
    drop(index);
    assert_eq!(drops.get(), 33);

    #[derive(Debug)]
    struct Key {
        value: usize,
        fail: Rc<Cell<bool>>,
    }
    impl PartialEq for Key {
        fn eq(&self, other: &Self) -> bool {
            self.value == other.value
        }
    }
    impl Eq for Key {}
    impl PartialOrd for Key {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Key {
        fn cmp(&self, other: &Self) -> Ordering {
            if self.fail.replace(false) {
                panic!("comparison failed")
            }
            self.value.cmp(&other.value)
        }
    }
    let fail = Rc::new(Cell::new(false));
    let mut index = PreparedIndex::default();
    let node = |key| {
        PreparedIndexNode::new(
            Key {
                value: key,
                fail: fail.clone(),
            },
            Value {
                lock: lock.clone(),
                drops: drops.clone(),
                value: key,
            },
            (),
        )
    };
    let mut first = Some(node(1));
    assert!(index.insert(&mut first));
    let mut incoming = Some(node(2));
    let address = std::ptr::from_ref(incoming.as_ref().unwrap().value());
    let held = lock.lock().unwrap();
    fail.set(true);
    assert!(catch_unwind(AssertUnwindSafe(|| index.insert(&mut incoming))).is_err());
    assert_eq!(index.len(), 1);
    assert_eq!(drops.get(), 33);
    assert_eq!(
        std::ptr::from_ref(incoming.as_ref().unwrap().value()),
        address
    );
    assert_eq!(index.get_by(|key| 1.cmp(&key.value)).unwrap().value, 1);
    drop(held);
    drop(incoming);
    assert_eq!(drops.get(), 34);
    drop(index);
    assert_eq!(drops.get(), 35);
}

#[test]
fn detached_provider_panic_retires_remaining_nodes_iteratively() {
    #[derive(Debug)]
    struct Value {
        drops: Rc<Cell<usize>>,
        panic_once: Rc<Cell<bool>>,
    }
    impl Drop for Value {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
            if self.panic_once.replace(false) {
                panic!("provider retirement");
            }
        }
    }
    let drops = Rc::new(Cell::new(0));
    let panic_once = Rc::new(Cell::new(false));
    let mut index = PreparedIndex::default();
    for key in 0..4096 {
        let mut candidate = Some(PreparedIndexNode::new(
            key,
            Value {
                drops: drops.clone(),
                panic_once: panic_once.clone(),
            },
            (),
        ));
        assert!(index.insert(&mut candidate));
    }
    let retired = index.extract_if(|_, _| true);
    assert!(index.is_empty());
    panic_once.set(true);
    assert!(catch_unwind(AssertUnwindSafe(|| drop(retired))).is_err());
    assert_eq!(drops.get(), 4096);
}

#[test]
fn equal_replacement_preserves_other_addresses_and_refused_candidate_ownership() {
    let mut index = PreparedIndex::default();
    let mut pointers = BTreeMap::new();
    for key in 0..127 {
        let node = PreparedIndexNode::new(key, key * 2, ());
        pointers.insert(key, std::ptr::from_ref(node.value()));
        assert!(index.insert(&mut Some(node)));
    }
    let mut candidate = Some(PreparedIndexNode::new(63, 999, ()));
    let candidate_pointer = std::ptr::from_ref(candidate.as_ref().unwrap().value());
    assert!(index.insert_or_replace(&mut candidate, |_| false).is_err());
    assert_eq!(
        candidate_pointer,
        std::ptr::from_ref(candidate.as_ref().unwrap().value())
    );
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        index.insert_or_replace(&mut candidate, |_| panic!("replacement observation"))
    }));
    assert!(panic.is_err());
    assert_eq!(
        candidate_pointer,
        std::ptr::from_ref(candidate.as_ref().unwrap().value())
    );
    let retired = index
        .insert_or_replace(&mut candidate, |value| *value == 126)
        .unwrap()
        .unwrap();
    assert!(candidate.is_none());
    assert_eq!(*retired.value(), 126);
    assert_eq!(index.len(), 127);
    for key in 0..127 {
        let value = index.get_by(|stored| key.cmp(stored)).unwrap();
        assert_eq!(*value, if key == 63 { 999 } else { key * 2 });
        assert_eq!(
            std::ptr::from_ref(value),
            if key == 63 {
                candidate_pointer
            } else {
                pointers[&key]
            }
        );
    }
    // The returned old node is child-free: dropping it cannot retire neighbors.
    drop(retired);
    assert_eq!(index.len(), 127);
    assert_eq!(*index.get_by(|stored| 62.cmp(stored)).unwrap(), 124);
}

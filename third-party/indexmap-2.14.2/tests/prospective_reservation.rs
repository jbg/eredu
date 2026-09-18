use indexmap::{IndexMap, IndexSet, TryReserveWithError};

#[test]
fn actual_backing_is_reported_before_allocation_and_reused_without_calls() {
    let mut map = IndexMap::<String, usize>::new();
    let mut paid = Vec::new();
    map.try_reserve_with(9, |layout| { paid.push(layout.size()); Ok::<_, ()>(()) }).unwrap();
    assert_eq!(paid.len(), 2);
    assert_eq!(paid.iter().sum::<usize>(), map.allocation_size());
    for index in 0..9 { map.insert(format!("key-{index}"), index); }
    let bytes = map.allocation_size();
    map.clear();
    map.try_reserve_with(9, |_| -> Result<(), ()> { panic!("retained backing must be reused") }).unwrap();
    assert_eq!(map.allocation_size(), bytes);
}

#[test]
fn refusal_retains_the_original_index_prefix_and_existing_order() {
    for stop in 0..2 {
        let mut map = IndexMap::<usize, usize>::new();
        let mut calls = 0;
        let result = map.try_reserve_with(7, |_| {
            let current = calls;
            calls += 1;
            if current == stop { Err(current) } else { Ok(()) }
        });
        assert!(matches!(result, Err(TryReserveWithError::Funding(index)) if index == stop));
        assert_eq!(calls, stop + 1);
        assert!(map.is_empty());
        assert_eq!(map.allocation_size() == 0, stop == 0);
        let mut retried = 0;
        map.try_reserve_with(7, |_| { retried += 1; Ok::<_, ()>(()) }).unwrap();
        assert_eq!(retried, 2 - stop);
        for key in [7, 3, 9, 1] { map.insert(key, key * 2); }
        map.insert(3, 70);
        assert_eq!(map.iter().map(|(&k, &v)| (k,v)).collect::<Vec<_>>(), [(7,14), (3,70), (9,18), (1,2)]);
        assert_eq!(map.iter().rev().map(|(&k, _)| k).collect::<Vec<_>>(), [1,9,3,7]);
    }
}

#[test]
fn funded_growth_preserves_replacement_and_removal_order() {
    let mut map = IndexMap::new();
    let mut reference = Vec::<(u32,u32)>::new();
    let mut seed = 19u32;
    for value in 0..4096 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let key = seed % 113;
        if value % 7 == 0 {
            let expected = reference.iter().position(|(k,_)| *k == key).map(|index| reference.remove(index).1);
            assert_eq!(map.shift_remove(&key), expected);
        } else {
            if !map.contains_key(&key) { map.try_reserve_with(1, |_| Ok::<_, ()>(())).unwrap(); }
            let expected = match reference.iter_mut().find(|(k,_)| *k == key) {
                Some((_, prior)) => Some(std::mem::replace(prior, value)),
                None => { reference.push((key,value)); None },
            };
            assert_eq!(map.insert(key,value), expected);
        }
        assert!(map.iter().map(|(&k,&v)| (k,v)).eq(reference.iter().copied()));
    }
    let mut set = IndexSet::new();
    set.try_reserve_with(3, |_| Ok::<_, ()>(())).unwrap();
    for key in [3,1,2,1] { set.insert(key); }
    assert_eq!(set.into_iter().collect::<Vec<_>>(), [3,1,2]);
}

#[test]
fn impossible_capacity_refuses_without_callback_or_mutation() {
    let mut map = IndexMap::<u64,u64>::new();
    let error = map.try_reserve_with(usize::MAX, |_| -> Result<(), ()> { panic!("overflow has no allocation") });
    assert!(matches!(error, Err(TryReserveWithError::Allocation(_))));
    assert_eq!(map.allocation_size(),0);
}

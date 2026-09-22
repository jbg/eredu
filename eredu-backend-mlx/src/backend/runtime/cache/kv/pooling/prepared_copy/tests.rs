use super::*;
use crate::backend::runtime::cache::kv::PoolingCacheState;
use safemlx::{Device, DeviceType, ops::indexing::TryIndexOp};
use std::{
    cell::Cell,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

fn payload(batch: i32, positions: i32, width: i32, seed: f32) -> Array {
    Array::from_slice(
        &(0..batch * positions * width)
            .map(|n| seed + n as f32 * 0.25)
            .collect::<Vec<_>>(),
        &[batch, positions, width],
    )
}

fn strided(positions: i32, seed: f32, stream: &Stream) -> Array {
    payload(2, 3, positions, seed)
        .transpose_axes(&[0, 2, 1], stream)
        .unwrap()
}

fn values(array: &Array, stream: &Stream) -> Vec<f32> {
    array
        .contiguous(false, stream)
        .unwrap()
        .evaluated()
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap()
}

fn snapshot(cache: &PoolingCache, stream: &Stream) -> [Option<Vec<f32>>; 5] {
    cache
        .state_arrays()
        .map(|slot| slot.map(|array| values(array, stream)))
}

fn populated(stream: &Stream) -> PoolingCache {
    let mut cache = PoolingCache::new(4).unwrap();
    let windows = cache
        .accumulate_windows(strided(9, 1., stream), strided(9, 101., stream), 0, stream)
        .unwrap();
    assert_eq!(windows.base_position, 0);
    assert_eq!(windows.values.shape(), [2, 8, 3]);
    cache
        .update_and_fetch(strided(2, 201., stream), stream)
        .unwrap();
    cache.replace_overlap(
        windows
            .values
            .try_index_device((.., 4..8, ..), stream)
            .unwrap(),
        windows
            .gates
            .try_index_device((.., 4..8, ..), stream)
            .unwrap(),
    );
    cache
}

fn advance(cache: &mut PoolingCache, seed: f32, stream: &Stream) {
    let windows = cache
        .accumulate_windows(
            strided(3, seed, stream),
            strided(3, seed + 100., stream),
            9,
            stream,
        )
        .unwrap();
    assert_eq!(windows.base_position, 8);
    cache
        .update_and_fetch(strided(1, seed + 200., stream), stream)
        .unwrap();
    cache.replace_overlap(windows.values, windows.gates);
}

thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.set(HOUSEKEEPING.get() + 1);
}
struct HousekeepingGuard;
impl Drop for HousekeepingGuard {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

#[test]
fn preparation_preserves_optional_slots_and_visits_exact_operands_without_native_work() {
    let stream = stream();
    let empty = PoolingCache::new(4).unwrap();
    let mut partial = PoolingCache::new(4).unwrap();
    partial
        .accumulate_windows(
            strided(3, 1., &stream),
            strided(3, 11., &stream),
            0,
            &stream,
        )
        .unwrap();
    let mut ready = PoolingCache::new(4).unwrap();
    ready
        .accumulate_windows(
            strided(4, 1., &stream),
            strided(4, 11., &stream),
            0,
            &stream,
        )
        .unwrap();
    // Accumulation precedes the architecture's pooling output. This valid
    // intermediate frontier must not acquire an invented pooled slot or fail.
    assert_eq!(ready.processed_tokens(), 4);
    assert_eq!(ready.pooled_tokens(), 0);
    let mut pooled = ready.clone();
    pooled
        .update_and_fetch(strided(1, 21., &stream), &stream)
        .unwrap();
    let mut overlap = PoolingCache::new(4).unwrap();
    overlap.replace_overlap(strided(4, 31., &stream), strided(4, 41., &stream));
    let complete = populated(&stream);

    for source in [empty, partial, ready, pooled, overlap, complete] {
        let expected_slots = source.state_arrays();
        let expected = expected_slots.map(|slot| slot.is_some());
        safemlx::register_thread_runtime_housekeeping(housekeeping);
        let guard = HousekeepingGuard;
        HOUSEKEEPING.set(0);
        let plan = source.prepare_isolated_copy();
        let mut operands = Vec::new();
        plan.visit_operands(&mut |array| operands.push(array));
        assert_eq!(
            operands.len(),
            expected.iter().filter(|present| **present).count()
        );
        for (actual, original) in operands.iter().zip(expected_slots.into_iter().flatten()) {
            assert!(std::ptr::eq(*actual, original));
        }
        assert_eq!(HOUSEKEEPING.get(), 0);
        drop(guard);
        let roots = RefCell::new(Vec::new());
        let copy = plan.copy_retained(&stream, &roots).unwrap();
        assert_eq!(copy.state_arrays().map(|slot| slot.is_some()), expected);
        assert_eq!(copy.ratio(), source.ratio());
        assert_eq!(copy.processed_tokens(), source.processed_tokens());
        assert_eq!(snapshot(&copy, &stream), snapshot(&source, &stream));
        assert_eq!(roots.borrow().len(), operands.len() * 2);
    }
}

#[test]
fn prepared_pooling_copy_preserves_partial_overlap_values_and_independent_continuation() {
    let stream = stream();
    let source = populated(&stream);
    let before = snapshot(&source, &stream);
    let source_info = source
        .state_arrays()
        .map(|slot| slot.unwrap().allocation_info().unwrap().unwrap());
    let roots = RefCell::new(Vec::new());
    let mut copy = source
        .prepare_isolated_copy()
        .copy_retained(&stream, &roots)
        .unwrap();
    assert_eq!(roots.borrow().len(), 10);
    assert_eq!(copy.ratio(), 4);
    assert_eq!(copy.processed_tokens(), 9);
    assert_eq!(copy.pooled_tokens(), 2);
    assert_eq!(snapshot(&copy, &stream), before);
    for (index, destination) in copy.state_arrays().into_iter().enumerate() {
        let destination = destination.unwrap();
        assert_eq!(
            destination.shape(),
            source.state_arrays()[index].unwrap().shape()
        );
        assert_ne!(
            destination.allocation_info().unwrap().unwrap().identity(),
            source_info[index].identity()
        );
    }
    let ordinary = source.deep_clone_state(&stream).unwrap();
    assert_eq!(snapshot(&ordinary, &stream), before);
    let mut oracle = populated(&stream);
    advance(&mut copy, 301., &stream);
    advance(&mut oracle, 301., &stream);
    assert_eq!(copy.processed_tokens(), 12);
    assert_eq!(copy.pooled_tokens(), 3);
    assert!(copy.pending_values.is_none());
    assert!(copy.pending_gates.is_none());
    assert_eq!(snapshot(&copy, &stream), snapshot(&oracle, &stream));
    assert_eq!(source.processed_tokens(), 9);
    assert_eq!(snapshot(&source, &stream), before);
    assert_eq!(snapshot(&ordinary, &stream), before);
}

#[test]
fn aliased_five_slot_sources_produce_independent_destinations_retained_by_the_collector() {
    let stream = stream();
    let backing = payload(2, 9, 3, 3.);
    let view = backing.try_index_device((.., 2..3, ..), &stream).unwrap();
    let expected = values(&view, &stream);
    let source_info = view.allocation_info().unwrap().unwrap();
    let mut source = PoolingCache::new(2).unwrap();
    source
        .restore_state(
            PoolingCacheState {
                pending_values: Some(view.clone()),
                pending_gates: Some(view.clone()),
                pooled: Some(view.clone()),
                overlap_values: Some(view.clone()),
                overlap_gates: Some(view),
            },
            3,
        )
        .unwrap();
    let roots = RefCell::new(Vec::new());
    let copy = source
        .prepare_isolated_copy()
        .copy_retained(&stream, &roots)
        .unwrap();
    assert_eq!(roots.borrow().len(), 10);
    let destinations = copy
        .arrays()
        .map(|array| array.allocation_info().unwrap().unwrap().identity())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(destinations.len(), 5);
    assert!(!destinations.contains(&source_info.identity()));

    #[derive(Debug)]
    struct Retired(Arc<AtomicUsize>);
    impl Drop for Retired {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let retired = Arc::new(AtomicUsize::new(0));
    for (slot, array) in copy.arrays().enumerate() {
        assert_eq!(
            array.allocation_info().unwrap().unwrap(),
            roots.borrow()[slot * 2 + 1]
                .allocation_info()
                .unwrap()
                .unwrap()
        );
        array
            .retain_allocation_owner(Retired(retired.clone()))
            .unwrap();
    }
    drop((copy, source, backing));
    safemlx::memory::clear_cache().unwrap();
    safemlx::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    for root in roots.borrow().iter() {
        assert_eq!(values(root, &stream), expected);
    }
    roots.borrow_mut().clear();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::memory::clear_cache().unwrap();
        safemlx::reclaim_allocation_owners();
        retired.load(Ordering::SeqCst) == 5
    });
}

#[test]
fn present_zero_length_slots_remain_present_in_the_copy() {
    let stream = stream();
    let empty = payload(2, 0, 3, 1.);
    let mut source = PoolingCache::new(4).unwrap();
    source
        .restore_state(
            PoolingCacheState {
                pending_values: Some(empty.clone()),
                pending_gates: Some(empty.clone()),
                pooled: Some(empty.clone()),
                overlap_values: Some(empty.clone()),
                overlap_gates: Some(empty),
            },
            0,
        )
        .unwrap();
    let copy = source.prepare_isolated_copy().copy(&stream).unwrap();
    assert_eq!(copy.ratio(), 4);
    assert_eq!(copy.processed_tokens(), 0);
    assert!(
        copy.state_arrays()
            .into_iter()
            .all(|slot| { slot.is_some_and(|array| array.shape() == [2, 0, 3]) })
    );
}

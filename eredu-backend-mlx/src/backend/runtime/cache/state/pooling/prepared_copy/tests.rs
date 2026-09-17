use super::*;
use crate::{backend::runtime::cache::residency::CacheResidencyManager, MlxTensor};
use eredu_nn::PoolingAttentionCache as _;
use eredu_runtime::PagedCacheOptions;
use safemlx::{Device, DeviceType};
use std::{cell::Cell, error::Error};

fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

fn payload(positions: i32, seed: f32) -> Array {
    Array::from_slice(
        &(0..2 * positions * 3)
            .map(|index| seed + index as f32 * 0.25)
            .collect::<Vec<_>>(),
        &[2, positions, 3],
    )
}

fn fill_pool(pool: &mut PoolingCache, seed: f32, stream: &Stream) {
    let windows = pool
        .accumulate_windows(payload(9, seed), payload(9, seed + 100.), 0, stream)
        .unwrap();
    pool.update_and_fetch(payload(9 / pool.ratio(), seed + 200.), stream)
        .unwrap();
    pool.replace_overlap(windows.values, windows.gates);
}

fn fixture(streams: usize, populated: bool, stream: &Stream) -> MlxPoolingAttentionCache {
    let ratios: &[i32] = match streams {
        0 => &[],
        1 => &[4],
        2 => &[4, 6],
        _ => unreachable!(),
    };
    let mut cache = MlxPoolingAttentionCache::with_streams(
        LiveKeyValueCache::resident(ConcatKeyValueCache::new_for_sliding_attention(7)),
        ratios,
    )
    .unwrap();
    if populated {
        cache
            .append_local(MlxTensor::from_array(payload(9, 1.)), stream)
            .unwrap();
        match &mut cache {
            MlxPoolingAttentionCache::Local(_) => {}
            MlxPoolingAttentionCache::Compressed { pool, .. } => fill_pool(pool, 11., stream),
            MlxPoolingAttentionCache::Sparse {
                pool, index_pool, ..
            } => {
                fill_pool(pool, 11., stream);
                fill_pool(index_pool, 31., stream);
            }
        }
    }
    cache
}

fn arrays(cache: &MlxPoolingAttentionCache) -> Vec<&Array> {
    let mut arrays = Vec::new();
    match cache.local() {
        LiveKeyValueCache::Resident(local) => arrays.extend(local.arrays()),
        LiveKeyValueCache::Paged(_) => panic!("resident fixture required"),
    }
    match cache {
        MlxPoolingAttentionCache::Local(_) => {}
        MlxPoolingAttentionCache::Compressed { pool, .. } => arrays.extend(pool.arrays()),
        MlxPoolingAttentionCache::Sparse {
            pool, index_pool, ..
        } => {
            arrays.extend(pool.arrays());
            arrays.extend(index_pool.arrays());
        }
    }
    arrays
}

fn values(cache: &MlxPoolingAttentionCache, stream: &Stream) -> Vec<Vec<f32>> {
    arrays(cache)
        .into_iter()
        .map(|array| {
            array
                .contiguous(false, stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap()
        })
        .collect()
}

fn metadata(cache: &MlxPoolingAttentionCache) -> Vec<(i32, i32, [bool; 5])> {
    let describe = |pool: &PoolingCache| {
        (
            pool.ratio(),
            pool.processed_tokens(),
            pool.state_arrays().map(|slot| slot.is_some()),
        )
    };
    match cache {
        MlxPoolingAttentionCache::Local(_) => Vec::new(),
        MlxPoolingAttentionCache::Compressed { pool, .. } => vec![describe(pool)],
        MlxPoolingAttentionCache::Sparse {
            pool, index_pool, ..
        } => {
            vec![describe(pool), describe(index_pool)]
        }
    }
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
fn resident_combinations_preserve_representation_slot_order_and_existing_worker_results() {
    let stream = stream();
    for streams in 0..=2 {
        for populated in [false, true] {
            let source = fixture(streams, populated, &stream);
            let expected = values(&source, &stream);
            let source_arrays = arrays(&source);
            safemlx::register_thread_runtime_housekeeping(housekeeping);
            let guard = HousekeepingGuard;
            HOUSEKEEPING.set(0);
            let plan = source.prepare_isolated_copy().unwrap();
            let mut operands = Vec::new();
            plan.visit_operands(&mut |array| operands.push(array));
            assert_eq!(operands.len(), if populated { 2 + 5 * streams } else { 0 });
            for (actual, expected) in operands.iter().zip(&source_arrays) {
                assert!(std::ptr::eq(*actual, *expected));
            }
            assert_eq!(HOUSEKEEPING.get(), 0);
            drop(guard);
            let roots = RefCell::new(Vec::new());
            let copy = plan.copy_retained(&stream, &roots).unwrap();
            assert_eq!(roots.borrow().len(), operands.len() * 2);
            assert_eq!(
                std::mem::discriminant(&copy),
                std::mem::discriminant(&source)
            );
            assert_eq!(copy.offset(), source.offset());
            assert_eq!(metadata(&copy), metadata(&source));
            assert_eq!(values(&copy, &stream), expected);
            let ordinary = source.isolated_snapshot(&stream).unwrap();
            assert_eq!(values(&ordinary, &stream), expected);
            assert_eq!(metadata(&ordinary), metadata(&copy));
            for (index, destination) in arrays(&copy).into_iter().enumerate() {
                let info = destination.allocation_info().unwrap().unwrap();
                assert_ne!(
                    info.identity(),
                    source_arrays[index]
                        .allocation_info()
                        .unwrap()
                        .unwrap()
                        .identity()
                );
                assert_eq!(
                    info,
                    roots.borrow()[2 * index + 1]
                        .allocation_info()
                        .unwrap()
                        .unwrap()
                );
            }
        }
    }
}

#[test]
fn shared_primary_and_index_sources_remain_independent_destination_streams() {
    let stream = stream();
    let mut source = fixture(2, true, &stream);
    let MlxPoolingAttentionCache::Sparse {
        pool, index_pool, ..
    } = &mut source
    else {
        unreachable!();
    };
    *index_pool = pool.clone();
    let expected = values(&source, &stream);
    let roots = RefCell::new(Vec::new());
    let mut copy = source
        .prepare_isolated_copy()
        .unwrap()
        .copy_retained(&stream, &roots)
        .unwrap();
    assert_eq!(roots.borrow().len(), 24);
    let MlxPoolingAttentionCache::Sparse {
        pool, index_pool, ..
    } = &mut copy
    else {
        panic!("sparse representation changed");
    };
    for (primary, index) in pool.arrays().zip(index_pool.arrays()) {
        assert_ne!(
            primary.allocation_info().unwrap().unwrap().identity(),
            index.allocation_info().unwrap().unwrap().identity()
        );
    }
    let prior_index = index_pool.processed_tokens();
    pool.accumulate_windows(payload(3, 301.), payload(3, 401.), 9, &stream)
        .unwrap();
    pool.update_and_fetch(payload(1, 501.), &stream).unwrap();
    assert_eq!(pool.processed_tokens(), 12);
    assert_eq!(index_pool.processed_tokens(), prior_index);
    assert_eq!(values(&source, &stream), expected);
    assert_ne!(values(&copy, &stream), expected);
}

#[test]
fn resident_composite_preparation_rejects_paged_local_storage_without_native_work() {
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1).unwrap())
            .unwrap();
    let source = MlxPoolingAttentionCache::with_streams(
        LiveKeyValueCache::paged_key_only(manager.clone(), 0, Some(7), 0, None).unwrap(),
        &[4, 6],
    )
    .unwrap();
    let identity = source.residency_manager().unwrap().session_id();
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let guard = HousekeepingGuard;
    HOUSEKEEPING.set(0);
    let error = match source.prepare_isolated_copy() {
        Ok(_) => panic!("paged local cache acquired a resident copy plan"),
        Err(error) => error,
    };
    assert!(error
        .source()
        .unwrap()
        .downcast_ref::<PagedPoolingCopy>()
        .is_some());
    assert_eq!(HOUSEKEEPING.get(), 0);
    assert_eq!(source.offset(), 0);
    assert_eq!(source.residency_manager().unwrap().session_id(), identity);
    drop(guard);
    let copied = source.deep_clone_state(&stream()).unwrap();
    assert_eq!(copied.residency_manager().unwrap().session_id(), identity);
    assert_eq!(metadata(&copied), metadata(&source));
}

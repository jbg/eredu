use super::*;
use crate::backend::runtime::cache::residency::CacheResidencyManager;
use crate::composition::MlxNeuralBackend;
use eredu_runtime::PagedCacheOptions;
use safemlx::{Device, DeviceType, ops::indexing::TryIndexOp};
use std::{
    cell::Cell,
    error::Error,
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

fn values(array: &Array, stream: &Stream) -> Vec<f32> {
    array
        .contiguous(false, stream)
        .unwrap()
        .evaluated()
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap()
}

fn fields(cache: &CompressedLatentCache) -> (i32, i32, i32, i32) {
    (cache.offset, cache.length, cache.capacity, cache.step)
}

fn operands<'a>(plan: &PreparedCompressedCopy<'a>) -> Vec<&'a Array> {
    let mut arrays = Vec::new();
    plan.visit_operands(&mut |array| arrays.push(array));
    arrays
}

fn invalid(cache: &CompressedLatentCache) -> Exception {
    match cache.prepare_isolated_copy() {
        Ok(_) => panic!("invalid resident cache was accepted"),
        Err(error) => error,
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
fn empty_and_zero_length_resident_copies_preserve_the_valid_slot_contract() {
    let stream = stream();
    let empty = CompressedLatentCache::new();
    let plan = empty.prepare_isolated_copy().unwrap();
    assert!(operands(&plan).is_empty());
    let roots = RefCell::new(Vec::new());
    let copy = plan.copy_retained(&stream, &roots).unwrap();
    assert_eq!(fields(&copy), fields(&empty));
    assert!(copy.arrays().is_none());
    assert!(roots.borrow().is_empty());

    let mut zero = CompressedLatentCache::new();
    zero.update_and_fetch(payload(2, 0, 4, 1.), payload(2, 0, 2, 7.), &stream)
        .unwrap();
    assert_eq!(fields(&zero), (0, 0, 0, zero.step));
    let plan = zero.prepare_isolated_copy().unwrap();
    let arrays = operands(&plan);
    assert_eq!(arrays.len(), 2);
    assert_eq!(arrays[0].shape(), [2, 0, 4]);
    assert_eq!(arrays[1].shape(), [2, 0, 2]);
    let copy = plan.copy(&stream).unwrap();
    assert_eq!(fields(&copy), fields(&zero));
    assert_eq!(copy.latent.as_ref().unwrap().shape(), [2, 0, 4]);
    assert_eq!(copy.rotary_key.as_ref().unwrap().shape(), [2, 0, 2]);
}

#[test]
fn prepared_resident_copy_rejects_incomplete_and_paged_storage_before_native_work() {
    let stream = stream();
    let mut source = CompressedLatentCache::new();
    source
        .update_and_fetch(payload(2, 3, 4, 1.), payload(2, 3, 2, 7.), &stream)
        .unwrap();
    let before = fields(&source);
    let source_latent = source.latent.as_ref().unwrap() as *const Array;
    let mut cases = Vec::new();
    for missing in 0..4 {
        let mut cache = source.clone();
        match missing {
            0 => cache.latent_storage = None,
            1 => cache.rotary_key_storage = None,
            2 => cache.latent = None,
            _ => cache.rotary_key = None,
        }
        cases.push(cache);
    }
    let manager = CacheResidencyManager::new(
        PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let paged = CompressedLatentCache::new_paged(manager, 0, None).unwrap();
    let mut bad_step = source.clone();
    bad_step.step = 0;
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let guard = HousekeepingGuard;
    HOUSEKEEPING.set(0);
    for cache in &cases {
        let error = invalid(cache);
        assert!(matches!(
            error
                .source()
                .unwrap()
                .downcast_ref::<InvalidResidentCopy>(),
            Some(InvalidResidentCopy::Inventory)
        ));
    }
    let error = invalid(&paged);
    assert!(matches!(
        error
            .source()
            .unwrap()
            .downcast_ref::<InvalidResidentCopy>(),
        Some(InvalidResidentCopy::Paged)
    ));
    let error = invalid(&bad_step);
    assert!(matches!(
        error
            .source()
            .unwrap()
            .downcast_ref::<InvalidResidentCopy>(),
        Some(InvalidResidentCopy::Frontier)
    ));
    let plan = source.prepare_isolated_copy().unwrap();
    let mut count = 0;
    plan.visit_operands(&mut |_| count += 1);
    assert_eq!(count, 2);
    assert_eq!(HOUSEKEEPING.get(), 0);
    assert_eq!(fields(&source), before);
    assert_eq!(
        source.latent.as_ref().unwrap() as *const Array,
        source_latent
    );
    drop(guard);
    // The separate paged isolated worker is still available, without treating
    // the resident plan as a proof for its manager or block storage.
    assert!(paged.isolated_snapshot(&stream).unwrap().is_paged());
}

#[test]
fn prepared_copy_uses_strided_logical_views_and_discards_resident_padding() {
    let stream = stream();
    let mut source = CompressedLatentCache::new();
    source
        .update_and_fetch(payload(2, 3, 4, 1.), payload(2, 3, 2, 101.), &stream)
        .unwrap();
    let before = fields(&source);
    assert!(source.capacity > source.length);
    let expected = (
        values(source.latent.as_ref().unwrap(), &stream),
        values(source.rotary_key.as_ref().unwrap(), &stream),
    );
    let original = [
        source
            .latent
            .as_ref()
            .unwrap()
            .allocation_info()
            .unwrap()
            .unwrap(),
        source
            .rotary_key
            .as_ref()
            .unwrap()
            .allocation_info()
            .unwrap()
            .unwrap(),
    ];
    assert!(original[0].bytes() > source.latent.as_ref().unwrap().nbytes());
    let plan = source.prepare_isolated_copy().unwrap();
    let arrays = operands(&plan);
    assert!(std::ptr::eq(arrays[0], source.latent.as_ref().unwrap()));
    assert!(std::ptr::eq(arrays[1], source.rotary_key.as_ref().unwrap()));
    assert!(!std::ptr::eq(
        arrays[0],
        source.latent_storage.as_ref().unwrap()
    ));
    assert_eq!(arrays[0].shape(), [2, 3, 4]);
    assert_eq!(arrays[1].shape(), [2, 3, 2]);
    let roots = RefCell::new(Vec::new());
    let mut copy = plan.copy_retained(&stream, &roots).unwrap();
    assert_eq!(roots.borrow().len(), 4);
    assert_eq!(
        fields(&copy),
        (source.offset, source.length, source.length, source.step)
    );
    for (index, (storage, logical)) in [
        (
            copy.latent_storage.as_ref().unwrap(),
            copy.latent.as_ref().unwrap(),
        ),
        (
            copy.rotary_key_storage.as_ref().unwrap(),
            copy.rotary_key.as_ref().unwrap(),
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let actual = logical.allocation_info().unwrap().unwrap();
        assert_ne!(actual.identity(), original[index].identity());
        assert_eq!(actual, storage.allocation_info().unwrap().unwrap());
    }
    assert_eq!(values(copy.latent.as_ref().unwrap(), &stream), expected.0);
    assert_eq!(
        values(copy.rotary_key.as_ref().unwrap(), &stream),
        expected.1
    );
    let ordinary = source.isolated_snapshot(&stream).unwrap();
    assert_eq!(fields(&ordinary), fields(&copy));
    assert_eq!(
        values(ordinary.latent.as_ref().unwrap(), &stream),
        expected.0
    );
    copy.update_and_fetch(payload(2, 1, 4, 71.), payload(2, 1, 2, 171.), &stream)
        .unwrap();
    assert_eq!(copy.offset(), 4);
    assert_eq!(fields(&source), before);
    assert_eq!(values(source.latent.as_ref().unwrap(), &stream), expected.0);
    assert_eq!(
        values(source.rotary_key.as_ref().unwrap(), &stream),
        expected.1
    );
}

#[test]
fn aliased_logical_inputs_still_copy_twice_and_collector_retains_every_result() {
    let stream = stream();
    let backing = payload(2, 11, 3, 3.);
    let view = backing.try_index_device((.., 2..5, ..), &stream).unwrap();
    let expected = values(&view, &stream);
    let source_info = view.allocation_info().unwrap().unwrap();
    let mut source = CompressedLatentCache::new();
    source.restore_resident(view.clone(), view, 3).unwrap();
    let plan = source.prepare_isolated_copy().unwrap();
    assert_eq!(operands(&plan).len(), 2);
    let roots = RefCell::new(Vec::new());
    let copy = plan.copy_retained(&stream, &roots).unwrap();
    let latent = copy
        .latent
        .as_ref()
        .unwrap()
        .allocation_info()
        .unwrap()
        .unwrap();
    let rotary = copy
        .rotary_key
        .as_ref()
        .unwrap()
        .allocation_info()
        .unwrap()
        .unwrap();
    assert_ne!(latent.identity(), rotary.identity());
    assert_ne!(latent.identity(), source_info.identity());
    assert_ne!(rotary.identity(), source_info.identity());
    assert_eq!(roots.borrow().len(), 4);
    assert_eq!(
        roots.borrow()[1].allocation_info().unwrap().unwrap(),
        latent
    );
    assert_eq!(
        roots.borrow()[3].allocation_info().unwrap().unwrap(),
        rotary
    );
    #[derive(Debug)]
    struct Retired(Arc<AtomicUsize>);
    impl Drop for Retired {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let retired = Arc::new(AtomicUsize::new(0));
    for array in [
        copy.latent.as_ref().unwrap(),
        copy.rotary_key.as_ref().unwrap(),
    ] {
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
        retired.load(Ordering::SeqCst) == 2
    });
}

#[test]
fn retained_source_inventory_includes_detached_stores_without_adding_copy_operands() {
    use crate::backend::runtime::residency::storage::RetainedStorage;
    use eredu_runtime::RuntimeLayerState;
    use std::collections::BTreeMap;

    let stream = stream();
    let mut original = CompressedLatentCache::new();
    original
        .update_and_fetch(payload(2, 3, 4, 1.), payload(2, 3, 2, 101.), &stream)
        .unwrap();
    let detached = original.deep_clone_state().unwrap();
    for (source, count) in [(original, 2), (detached, 4)] {
        let mut expected = BTreeMap::new();
        for array in [
            source.latent_storage.as_ref().unwrap(),
            source.rotary_key_storage.as_ref().unwrap(),
            source.latent.as_ref().unwrap(),
            source.rotary_key.as_ref().unwrap(),
        ] {
            array.evaluated().unwrap();
            let info = array.allocation_info().unwrap().unwrap();
            expected.insert(info.identity(), info.bytes() as u64);
        }
        assert_eq!(expected.len(), count);
        let mut retained = RetainedStorage::default();
        for value in source.retained_values() {
            retained.include_array(value.as_array()).unwrap();
        }
        assert_eq!(retained.array_allocation_facts(), expected);
        assert_eq!(
            retained.byte_bound().unwrap(),
            Some(expected.values().sum())
        );

        let plan = source.prepare_isolated_copy().unwrap();
        assert_eq!(operands(&plan).len(), 2);
        let copy = plan.copy(&stream).unwrap();
        let mut copied = RetainedStorage::default();
        for value in copy.retained_values() {
            value.as_array().evaluated().unwrap();
            copied.include_array(value.as_array()).unwrap();
        }
        let copied_roots = copied.array_allocation_facts();
        assert_eq!(copied_roots.len(), 2);
        assert!(copied_roots.keys().all(|id| !expected.contains_key(id)));
        assert_eq!(
            values(copy.latent.as_ref().unwrap(), &stream),
            values(source.latent.as_ref().unwrap(), &stream)
        );
        assert_eq!(
            values(copy.rotary_key.as_ref().unwrap(), &stream),
            values(source.rotary_key.as_ref().unwrap(), &stream)
        );
    }
}

#[test]
fn borrowed_compressed_visit_keeps_all_backing_and_logical_fields_without_native_work() {
    use eredu_runtime::RuntimeLayerState;
    let stream = stream();
    let mut initial = CompressedLatentCache::new();
    initial
        .update_and_fetch(payload(2, 3, 4, 1.), payload(2, 3, 2, 7.), &stream)
        .unwrap();
    let independent = initial.deep_clone_state().unwrap();
    let compact = initial
        .prepare_isolated_copy()
        .unwrap()
        .copy(&stream)
        .unwrap();
    for cache in [&initial, &independent, &compact] {
        // Compare actual fields, including aliases, rather than logical copy operands.
        let expected = [
            cache.latent_storage.as_ref().unwrap(),
            cache.rotary_key_storage.as_ref().unwrap(),
            cache.latent.as_ref().unwrap(),
            cache.rotary_key.as_ref().unwrap(),
        ];
        let before = fields(cache);
        safemlx::register_thread_runtime_housekeeping(housekeeping);
        let guard = HousekeepingGuard;
        HOUSEKEEPING.set(0);
        let mut count = 0;
        RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(cache, &mut |value| {
            assert!(std::ptr::eq(value.as_array(), expected[count]));
            count += 1;
        });
        assert_eq!(count, 4);
        assert_eq!(HOUSEKEEPING.get(), 0);
        drop(guard);
        assert_eq!(fields(cache), before);
    }
    let empty = CompressedLatentCache::new();
    RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(&empty, &mut |_| {
        panic!("empty resident cache")
    });
}

#[test]
fn borrowed_paged_compressed_visit_covers_actual_tail_without_loading_sealed_blocks() {
    use eredu_runtime::RuntimeLayerState;
    let stream = stream();
    let manager = CacheResidencyManager::new(
        PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let mut cache = CompressedLatentCache::new_paged(manager, 0, None).unwrap();
    cache
        .update_and_fetch(payload(1, 5, 4, 1.), payload(1, 5, 2, 11.), &stream)
        .unwrap();
    let paged = cache.paged.as_deref().unwrap();
    let sealed = paged.block_ids().unwrap();
    assert_eq!(sealed.len(), 2);
    let expected = [
        paged.tail_latent.as_ref().unwrap(),
        paged.tail_rotary.as_ref().unwrap(),
    ];
    let before = (paged.offset, paged.tail_start);
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let guard = HousekeepingGuard;
    HOUSEKEEPING.set(0);
    let mut count = 0;
    RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(&cache, &mut |value| {
        assert!(std::ptr::eq(value.as_array(), expected[count]));
        count += 1;
    });
    assert_eq!(count, 2);
    assert_eq!(HOUSEKEEPING.get(), 0);
    drop(guard);
    assert_eq!((paged.offset, paged.tail_start), before);
    assert_eq!(paged.block_ids().unwrap(), sealed);
    // Manager-owned sealed/host blocks require their separate retained inventory.
}

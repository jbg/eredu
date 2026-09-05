#[test]
#[ignore = "requires MLX runtime execution"]
fn paged_key_only_cache_uses_canonical_sentinel_blocks() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(2, 4096, 4096, 1).unwrap()).unwrap();
    let mut cache =
        PagedKeyValueCache::new_key_only_with_layout(manager.clone(), 0, Some(8), 0, None).unwrap();
    let keys = Array::from_slice(
        &(0..20).map(|v| v as f32).collect::<Vec<_>>(),
        &[1, 1, 5, 4],
    );
    let values = zeros_dtype(&[1, 1, 5, 0], Dtype::Float32, stream).unwrap();
    let (visible_keys, visible_values) = cache.update_and_fetch(keys, values, stream).unwrap();
    assert_eq!(visible_keys.shape(), &[1, 1, 5, 4]);
    assert_eq!(visible_values.shape(), &[1, 1, 5, 1]);
    cache.finalize().unwrap();
    let ids = manager
        .layer_block_ids(0, CacheRepresentation::KeyValue, 0, 5, 0)
        .unwrap();
    assert_eq!(ids.len(), 3);
    for id in ids {
        let lease = manager.lease_block(&id, stream).unwrap();
        let CacheBlockArrays::KeyValue { keys, values } = lease.arrays() else {
            panic!("key-only cache emitted the wrong block representation");
        };
        assert_eq!(keys.dim(-1), 4);
        assert_eq!(values.dim(-1), 1);
    }
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn cache_checkpoint_clone_shares_kv_array_handles() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let mut cache = ConcatKeyValueCache::new_with_step(8);
    let keys = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 1, 2]);
    cache.update_and_fetch(keys.clone(), keys, stream).unwrap();
    let checkpoint = cache.clone();
    let cache_arrays = cache.arrays().collect::<Vec<_>>();
    let checkpoint_arrays = checkpoint.arrays().collect::<Vec<_>>();

    assert_eq!(cache_arrays.len(), 2);
    assert_eq!(checkpoint_arrays.len(), 2);
    for (current, saved) in cache_arrays.into_iter().zip(checkpoint_arrays) {
        eval([current, saved]).unwrap();
        let current = current.evaluated().unwrap();
        let saved = saved.evaluated().unwrap();
        assert_eq!(
            current.as_slice::<f32>().as_ptr(),
            saved.as_slice::<f32>().as_ptr()
        );
    }
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn pooling_cache_is_invariant_to_decode_chunking() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let mut cache = PoolingCache::new(4).unwrap();
    let first = cache
        .accumulate_windows(
            Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3, 1]),
            Array::from_slice(&[0.0f32; 3], &[1, 3, 1]),
            0,
            stream,
        )
        .unwrap();
    assert_eq!(first.values.dim(1), 0);
    let second = cache
        .accumulate_windows(
            Array::from_slice(&[4.0f32, 5.0], &[1, 2, 1]),
            Array::from_slice(&[0.0f32; 2], &[1, 2, 1]),
            3,
            stream,
        )
        .unwrap();
    assert_eq!(second.base_position, 0);
    assert_eq!(second.values.shape(), [1, 4, 1]);
    assert_eq!(cache.processed_tokens(), 5);
    assert_eq!(
        second.values.evaluated().unwrap().as_slice::<f32>(),
        &[1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn chunked_cache_truncate_preserves_backing_capacity_and_overwrites_rollback() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let mut cache = ConcatKeyValueCache::new_with_step(8);
    let initial = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 1, 3, 1]);
    cache
        .update_and_fetch(initial.clone(), initial, stream)
        .unwrap();
    eval(cache.arrays()).unwrap();
    let pointers_before = cache
        .arrays()
        .map(|array| array.evaluated().unwrap().as_slice::<f32>().as_ptr())
        .collect::<Vec<_>>();

    cache.truncate(1, stream).unwrap();

    assert_eq!(cache.length, 1);
    assert_eq!(cache.capacity, 8);
    assert!(cache.arrays().all(|array| array.dim(-2) == 8));
    let pointers_after = cache
        .arrays()
        .map(|array| array.evaluated().unwrap().as_slice::<f32>().as_ptr())
        .collect::<Vec<_>>();
    assert_eq!(pointers_after, pointers_before);

    let replacement = Array::from_slice(&[7.0f32, 8.0], &[1, 1, 2, 1]);
    let (keys, values) = cache
        .update_and_fetch(replacement.clone(), replacement, stream)
        .unwrap();
    eval([&keys, &values]).unwrap();
    assert_eq!(
        keys.evaluated().unwrap().as_slice::<f32>(),
        &[1.0, 7.0, 8.0]
    );
    assert_eq!(
        values.evaluated().unwrap().as_slice::<f32>(),
        &[1.0, 7.0, 8.0]
    );
    assert_eq!(cache.capacity, 8);
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn chunked_cache_grows_by_steps_and_preserves_sliding_values() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let mut cache = ConcatKeyValueCache::new_with_max_size_and_step(8, 4);

    let mut fetched = None;
    for value in 0..10 {
        let keys =
            Array::full::<f32>(&[1, 1, 1, 2], Array::from_f32(value as f32), stream).unwrap();
        let values = keys.clone();
        fetched = Some(cache.update_and_fetch(keys, values, stream).unwrap().0);
        if value == 0 {
            assert_eq!(cache.capacity, 4);
        } else if value == 4 {
            assert_eq!(cache.capacity, 8);
        }
    }

    let fetched = fetched.unwrap();
    assert_eq!(cache.offset(), 10);
    assert_eq!(fetched.shape(), &[1, 1, 8, 2]);
    assert_eq!(
        fetched
            .try_index_device((0, 0, 0, 0), stream)
            .unwrap()
            .item::<f32>(stream),
        2.0
    );
    assert_eq!(
        fetched
            .try_index_device((0, 0, -1, 0), stream)
            .unwrap()
            .item::<f32>(stream),
        9.0
    );
}

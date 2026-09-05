#[test]
#[ignore = "requires MLX runtime execution"]
fn paged_cache_enforces_one_budget_across_layers() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    // The finite device budget covers one protected block per layer plus
    // the active mutable tail.
    // Host-transfer buffers are page-aligned, so one demoted logical
    // block consumes one 32 KiB physical host allocation.
    let options = PagedCacheOptions::new(2, 48, 32 * 1024, 1)
        .unwrap()
        .with_full_attention(true);
    let manager = CacheResidencyManager::new(options).unwrap();
    let mut first = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
    let mut second = PagedKeyValueCache::new(manager.clone(), 1, None).unwrap();
    for cache in [&mut first, &mut second] {
        let states = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 1, 4, 1]);
        cache
            .update_and_fetch(states.clone(), states, stream)
            .unwrap();
    }
    let report = manager.report().unwrap();
    assert_eq!(report.key_value_blocks, 4);
    assert_eq!(report.device_blocks, 3);
    assert_eq!(report.host_blocks, 1);
    assert_eq!(report.current_device_bytes, 48);
    assert_eq!(report.current_host_bytes, 32 * 1024);
    assert!(report.peak_device_bytes <= manager.options().device_budget_bytes());
    assert!(report.peak_host_bytes <= manager.options().host_budget_bytes());
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn rejected_key_value_append_restores_tail_and_newly_sealed_blocks() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let options = PagedCacheOptions::new(2, 20, 4096, 1)
        .unwrap()
        .with_full_attention(true);
    let manager = CacheResidencyManager::new(options).unwrap();
    let mut cache = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
    cache
        .append(
            Array::from_slice(&[1.0f32], &[1, 1, 1, 1]),
            Array::from_slice(&[10.0f32], &[1, 1, 1, 1]),
            stream,
        )
        .unwrap();

    let error = cache
        .append(
            Array::from_slice(&[2.0f32, 3.0], &[1, 1, 2, 1]),
            Array::from_slice(&[20.0f32, 30.0], &[1, 1, 2, 1]),
            stream,
        )
        .expect_err("the protected sealed block plus tail must exceed the device budget");
    assert!(error.what().contains("budget exceeded"));
    assert_eq!(cache.offset, 1);
    assert_eq!(cache.tail_start, 0);
    assert_eq!(cache.tail_len(), 1);
    assert!(manager
        .layer_block_ids(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
        .unwrap()
        .is_empty());
    let report = manager.report().unwrap();
    assert_eq!(report.mutable_tail_bytes, 8);
    assert_eq!(report.current_device_bytes, 8);
    assert_eq!(report.logical_cached_tokens, 1);
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn rejected_compressed_append_restores_tail_and_newly_sealed_blocks() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let options = PagedCacheOptions::new(2, 20, 4096, 1)
        .unwrap()
        .with_full_attention(true);
    let manager = CacheResidencyManager::new(options).unwrap();
    let mut cache = CompressedLatentCache::new_paged(manager.clone(), 0, None).unwrap();
    cache
        .update_and_fetch(
            Array::from_slice(&[1.0f32], &[1, 1, 1]),
            Array::from_slice(&[10.0f32], &[1, 1, 1]),
            stream,
        )
        .unwrap();

    let error = cache
        .update_and_fetch(
            Array::from_slice(&[2.0f32, 3.0], &[1, 2, 1]),
            Array::from_slice(&[20.0f32, 30.0], &[1, 2, 1]),
            stream,
        )
        .expect_err("the protected sealed block plus tail must exceed the device budget");
    assert!(error.what().contains("budget exceeded"));
    assert_eq!(cache.offset(), 1);
    let tail = cache
        .paged_tail_block()
        .expect("the original compressed tail must be restored");
    assert_eq!((tail.start, tail.end), (0, 1));
    assert!(cache.paged_block_ids().unwrap().unwrap().is_empty());
    let report = manager.report().unwrap();
    assert_eq!(report.mutable_tail_bytes, 8);
    assert_eq!(report.current_device_bytes, 8);
    assert_eq!(report.logical_cached_tokens, 1);
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn paged_cache_truncates_at_and_inside_sealed_blocks() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let manager = CacheResidencyManager::new(paged_options(true)).unwrap();
    let mut cache = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
    let states = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0], &[1, 1, 5, 1]);
    cache
        .update_and_fetch(states.clone(), states, stream)
        .unwrap();

    cache.truncate(3, stream).unwrap();
    assert_eq!(cache.offset(), 3);
    assert_eq!(manager.report().unwrap().logical_cached_tokens, 3);
    let suffix = Array::from_slice(&[9.0f32], &[1, 1, 1, 1]);
    cache
        .update_and_fetch(suffix.clone(), suffix, stream)
        .unwrap();
    assert_eq!(cache.offset(), 4);

    cache.truncate(2, stream).unwrap();
    assert_eq!(cache.offset(), 2);
    assert_eq!(manager.report().unwrap().logical_cached_tokens, 2);
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn rejected_sealed_truncation_preserves_local_and_manager_state() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let manager = CacheResidencyManager::new(paged_options(true)).unwrap();
    let mut cache = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
    let states = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 7, 1]);
    cache
        .update_and_fetch(states.clone(), states, stream)
        .unwrap();
    let before_ids = manager
        .layer_block_ids(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
        .unwrap();
    let leased_id = before_ids.iter().find(|id| id.start == 2).unwrap().clone();
    let suffix_lease = manager.lease_block(&leased_id, stream).unwrap();

    let error = cache
        .truncate(1, stream)
        .expect_err("a leased middle suffix block must reject truncation");
    assert!(error.what().contains("has lease count 1, expected 0"));
    assert_eq!(cache.offset(), 7);
    assert_eq!(cache.tail_start, 6);
    assert_eq!(cache.tail_len(), 1);
    assert_eq!(
        cache
            .tail_keys
            .as_ref()
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>(),
        &[6.0]
    );
    assert_eq!(
        manager
            .layer_block_ids(0, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
            .unwrap(),
        before_ids
    );
    assert_eq!(manager.report().unwrap().logical_cached_tokens, 7);

    drop(suffix_lease);
    cache.truncate(1, stream).unwrap();
    assert_eq!(cache.offset(), 1);
    assert_eq!(manager.report().unwrap().logical_cached_tokens, 1);
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn compressed_latent_paging_seals_atomic_block_pairs() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let manager = CacheResidencyManager::new(paged_options(true)).unwrap();
    let mut cache = CompressedLatentCache::new_paged(manager.clone(), 0, None).unwrap();
    let latent = Array::from_slice(
        &[0.0f32, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1],
        &[1, 5, 2],
    );
    let rotary = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0], &[1, 5, 1]);
    cache.update_and_fetch(latent, rotary, stream).unwrap();
    assert_eq!(cache.offset(), 5);
    let before = manager.report().unwrap();
    assert_eq!(before.compressed_latent_blocks, 2);
    assert!(before.mutable_tail_bytes > 0);
    cache.finalize().unwrap();
    let after = manager.report().unwrap();
    assert_eq!(after.compressed_latent_blocks, 3);
    assert_eq!(after.mutable_tail_bytes, 0);
    assert_eq!(after.logical_cached_tokens, 5);
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn compressed_latent_host_demotion_and_rehydration_preserve_atomic_pairs() {
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = context.stream();
    // Each two-token latent/rotary pair occupies 24 logical bytes. Two
    // blocks fit on the device and one page-aligned host allocation fits
    // in the finite host tier.
    let options = PagedCacheOptions::new(2, 48, 32 * 1024, 1)
        .unwrap()
        .with_full_attention(true);
    let manager = CacheResidencyManager::new(options).unwrap();
    let mut cache = CompressedLatentCache::new_paged(manager.clone(), 0, None).unwrap();
    let latent_values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
    let rotary_values = (100..106).map(|value| value as f32).collect::<Vec<_>>();
    cache
        .update_and_fetch(
            Array::from_slice(&latent_values, &[1, 6, 2]),
            Array::from_slice(&rotary_values, &[1, 6, 1]),
            stream,
        )
        .unwrap();

    let report = manager.report().unwrap();
    assert_eq!(report.compressed_latent_blocks, 3);
    assert_eq!(report.device_blocks, 2);
    assert_eq!(report.host_blocks, 1);
    assert_eq!(report.current_device_bytes, 48);
    assert_eq!(report.current_host_bytes, 32 * 1024);
    assert_eq!(report.host_demotions, 1);
    let first = cache.paged_block_ids().unwrap().unwrap().remove(0);
    assert_eq!(
        first.representation,
        CacheRepresentation::CompressedLatentRotary
    );

    let lease = manager.lease_block(&first, stream).unwrap();
    match lease.arrays() {
        CacheBlockArrays::CompressedLatentRotary { latent, rotary_key } => {
            eval([latent, rotary_key]).unwrap();
            assert_eq!(
                latent.evaluated().unwrap().as_slice::<f32>(),
                &latent_values[..4]
            );
            assert_eq!(
                rotary_key.evaluated().unwrap().as_slice::<f32>(),
                &rotary_values[..2]
            );
        }
        CacheBlockArrays::KeyValue { .. } => {
            panic!("compressed-latent block was rehydrated as key/value state")
        }
    }
    drop(lease);
    let report = manager.report().unwrap();
    assert_eq!(report.host_promotions, 1);
    assert!(report.host_demotions >= 2);
    assert!(report.current_device_bytes <= manager.options().device_budget_bytes());
    assert!(report.current_host_bytes <= manager.options().host_budget_bytes());
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn compressed_latent_live_disk_demotion_and_rehydration_preserve_atomic_pairs() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let directory = tempfile::tempdir().unwrap();
    let options = PagedCacheOptions::new(2, 48, 32 * 1024, 1)
        .unwrap()
        .with_full_attention(true)
        .with_live_disk(directory.path(), 4096, 2)
        .unwrap();
    let manager = CacheResidencyManager::new(options).unwrap();
    let mut cache = CompressedLatentCache::new_paged(manager.clone(), 0, None).unwrap();
    let latent_values = (0..16).map(|value| value as f32).collect::<Vec<_>>();
    let rotary_values = (100..108).map(|value| value as f32).collect::<Vec<_>>();
    cache
        .update_and_fetch(
            Array::from_slice(&latent_values, &[1, 8, 2]),
            Array::from_slice(&rotary_values, &[1, 8, 1]),
            stream,
        )
        .unwrap();

    let mut report = manager.report().unwrap();
    for _ in 0..100 {
        if report.disk_blocks >= 1 && report.in_flight_write_blocks == 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
        report = manager.report().unwrap();
    }
    assert_eq!(report.compressed_latent_blocks, 4);
    assert!(report.disk_blocks >= 1);
    assert!(report.disk_demotions >= 1);
    assert!(report.current_device_bytes <= manager.options().device_budget_bytes());
    assert!(report.current_host_bytes <= manager.options().host_budget_bytes());
    let first = cache.paged_block_ids().unwrap().unwrap().remove(0);

    let lease = manager.lease_block(&first, stream).unwrap();
    match lease.arrays() {
        CacheBlockArrays::CompressedLatentRotary { latent, rotary_key } => {
            eval([latent, rotary_key]).unwrap();
            assert_eq!(
                latent.evaluated().unwrap().as_slice::<f32>(),
                &latent_values[..4]
            );
            assert_eq!(
                rotary_key.evaluated().unwrap().as_slice::<f32>(),
                &rotary_values[..2]
            );
        }
        CacheBlockArrays::KeyValue { .. } => {
            panic!("compressed-latent disk block was rehydrated as key/value state")
        }
    }
    drop(lease);
    let report = manager.report().unwrap();
    assert!(report.disk_promotions >= 1);
    assert!(report.current_device_bytes <= manager.options().device_budget_bytes());
    assert!(report.current_host_bytes <= manager.options().host_budget_bytes());

    drop(cache);
    drop(manager);
    assert!(fs::read_dir(directory.path()).unwrap().next().is_none());
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn paged_sliding_cache_discards_invisible_blocks_and_preserves_offsets() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let manager = CacheResidencyManager::new(paged_options(false)).unwrap();
    let mut cache = PagedKeyValueCache::new(manager.clone(), 0, Some(3)).unwrap();
    let prefix = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0], &[1, 1, 5, 1]);
    cache
        .update_and_fetch(prefix.clone(), prefix, stream)
        .unwrap();
    let next = Array::from_slice(&[5.0f32], &[1, 1, 1, 1]);
    let visible = cache
        .update_and_fetch(next.clone(), next, stream)
        .unwrap()
        .0;
    assert_eq!(cache.offset(), 6);
    assert_eq!(visible.shape(), &[1, 1, 3, 1]);
    assert_eq!(
        visible
            .try_index_device((0, 0, 0, 0), stream)
            .unwrap()
            .item::<f32>(stream),
        3.0
    );
    assert_eq!(manager.report().unwrap().discarded_sliding_blocks, 1);
}

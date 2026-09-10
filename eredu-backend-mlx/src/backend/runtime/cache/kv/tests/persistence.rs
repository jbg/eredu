#[test]
#[ignore = "requires MLX runtime execution"]
fn prompt_cache_is_atomic_inspectable_and_reopens_lazily() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let options = paged_options(true);
    let manager = CacheResidencyManager::new(options.clone()).unwrap();
    let rank = CacheRankIdentity::new(Some(1), None, None);
    let mut cache =
        PagedKeyValueCache::new_with_layout(manager.clone(), 0, None, 0, Some(rank)).unwrap();
    let states = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0], &[1, 1, 5, 1]);
    cache
        .update_and_fetch(states.clone(), states, stream)
        .unwrap();
    cache.finalize().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("prompt-cache");
    let descriptor = PromptCacheDescriptor::new(
        "decoder",
        "decoder",
        "sha256:test-checkpoint",
        "tokens:11,12,13,14,15",
        "sha256:test-architecture",
        1,
        0,
        1,
        1,
        PromptCacheModelIdentity::key_value_layouts([None], 1, 1).unwrap(),
        vec![0],
        vec![eredu_core::cache::PromptCacheStateSegment::new("state", 0..1).unwrap()],
        0,
        PromptCacheTopology::new(Some((2, 1)), None, None, true).unwrap(),
    )
    .unwrap();
    let tokens = [11u32, 12, 13, 14, 15];
    let invalid_destination = directory.path().join("invalid-prompt-cache");
    let invalid_descriptor = descriptor
        .clone()
        .with_topology(PromptCacheTopology::new(Some((2, 0)), None, None, true).unwrap())
        .unwrap();
    assert!(manager
        .save_prompt_cache(
            &invalid_destination,
            invalid_descriptor,
            &tokens,
            &[],
            &PromptCacheOptions::default(),
        )
        .is_err());
    assert!(!invalid_destination.exists());
    manager
        .save_prompt_cache(
            &destination,
            descriptor.clone(),
            &tokens,
            &[],
            &PromptCacheOptions::default(),
        )
        .unwrap();
    let inspected = inspect_prompt_cache(&destination).unwrap();
    assert_eq!(inspected.total_prefix_tokens, 5);
    assert!(inspected
        .blocks
        .iter()
        .all(|block| block.rank == Some(rank)));
    drop(cache);
    drop(manager);
    assert!(resolve_prompt_cache_root(&destination)
        .unwrap()
        .join("manifest.json")
        .is_file());

    let incompatible = descriptor
        .clone()
        .with_topology(PromptCacheTopology::new(Some((2, 0)), None, None, true).unwrap())
        .unwrap();
    let identity = PromptCacheModelIdentity::new(
        descriptor.model_family(),
        descriptor.effective_model_type(),
        descriptor.architecture_fingerprint(),
        1,
        0,
        1,
        0,
        descriptor.topology().clone(),
        descriptor.layer_layout().clone(),
        vec![0],
        descriptor.state_segments().to_vec(),
    )
    .unwrap();
    assert!(open_prompt_cache(
        &destination,
        &incompatible,
        &identity,
        &tokens,
        options.clone()
    )
    .is_err());

    let (loaded_manager, loaded_manifest) =
        open_prompt_cache(&destination, &descriptor, &identity, &tokens, options).unwrap();
    assert_eq!(loaded_manifest.blocks.len(), 3);
    let mut restored =
        PagedKeyValueCache::new_with_layout(loaded_manager.clone(), 0, None, 0, Some(rank))
            .unwrap();
    assert_eq!(restored.offset(), 5);
    let suffix = Array::from_slice(&[5.0f32], &[1, 1, 1, 1]);
    restored
        .update_and_fetch(suffix.clone(), suffix, stream)
        .unwrap();
    assert_eq!(restored.offset(), 6);
    assert_eq!(loaded_manager.report().unwrap().prompt_cache_loads, 1);

    let manifest_path = resolve_prompt_cache_root(&destination)
        .unwrap()
        .join("manifest.json");
    let mut corrupted: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    corrupted["blocks"][0]["logical_bytes"] = serde_json::json!(1);
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&corrupted).unwrap(),
    )
    .unwrap();
    assert!(inspect_prompt_cache(&destination).is_err());
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn distinct_window_schedule_save_reopen_continue_preserves_ranges() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let options = PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
        .unwrap()
        .with_full_attention(true)
        .with_persistence_retention(true);
    let manager = CacheResidencyManager::new(options.clone()).unwrap();
    let windows = [None, Some(3), Some(5), Some(2)];
    let mut caches = windows
        .iter()
        .enumerate()
        .map(|(layer, window)| PagedKeyValueCache::new(manager.clone(), layer, *window).unwrap())
        .collect::<Vec<_>>();
    let prefix = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 6, 1]);
    for cache in &mut caches {
        cache
            .update_and_fetch(prefix.clone(), prefix.clone(), stream)
            .unwrap();
        cache.finalize().unwrap();
    }
    let retained_before = windows
        .iter()
        .enumerate()
        .map(|(layer, _)| {
            manager
                .layer_block_ids(layer, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
                .unwrap()
                .into_iter()
                .map(|id| (id.start, id.end))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(retained_before, vec![vec![(0, 2), (2, 4), (4, 6)]; 4]);
    let layer_layout = PromptCacheModelIdentity::key_value_layouts(windows, 1, 1).unwrap();
    let descriptor = PromptCacheDescriptor::new(
        "schedule-fixture",
        "schedule-fixture",
        "checkpoint",
        "tokens:0..6",
        "architecture",
        windows.len(),
        0,
        windows.len(),
        1,
        layer_layout.clone(),
        vec![0; windows.len()],
        vec![eredu_core::cache::PromptCacheStateSegment::new("state", 0..windows.len()).unwrap()],
        0,
        PromptCacheTopology::default(),
    )
    .unwrap();
    let tokens = [10, 11, 12, 13, 14, 15];
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("distinct-windows");
    manager
        .save_prompt_cache(
            &destination,
            descriptor.clone(),
            &tokens,
            &[],
            &PromptCacheOptions::default(),
        )
        .unwrap();

    let query = Array::from_slice(&[1.0f32], &[1, 1, 1, 1]);
    let suffix = Array::from_slice(&[6.0f32], &[1, 1, 1, 1]);
    let mut uninterrupted = Vec::new();
    for (cache, window) in caches.iter_mut().zip(windows) {
        cache
            .update_for_attention(suffix.clone(), suffix.clone(), stream)
            .unwrap();
        let output = cache
            .paged_attention(
                &query,
                1.0,
                None,
                None,
                None,
                eredu_nn::AttentionArithmetic::Fused,
                stream,
            )
            .unwrap()
            .unwrap()
            .evaluated()
            .unwrap()
            .item::<f32>();
        let expected_start = window.map_or(0, |window| 6 - i64::from(window - 1));
        let normalizer = (expected_start..=6)
            .map(|value| ((value - 6) as f32).exp())
            .sum::<f32>();
        let expected = (expected_start..=6)
            .map(|value| value as f32 * ((value - 6) as f32).exp())
            .sum::<f32>()
            / normalizer;
        assert!((output - expected).abs() < 1e-5);
        uninterrupted.push(output);
    }
    drop(caches);
    drop(manager);

    let identity = PromptCacheModelIdentity::new(
        descriptor.model_family(),
        descriptor.effective_model_type(),
        descriptor.architecture_fingerprint(),
        windows.len(),
        0,
        windows.len(),
        0,
        PromptCacheTopology::default(),
        layer_layout,
        vec![0; windows.len()],
        descriptor.state_segments().to_vec(),
    )
    .unwrap();
    let (manager, manifest) =
        open_prompt_cache(&destination, &descriptor, &identity, &tokens, options).unwrap();
    assert_eq!(&manifest.layer_layout, descriptor.layer_layout());

    for (layer, window) in windows.into_iter().enumerate() {
        let retained_after = manager
            .layer_block_ids(layer, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
            .unwrap()
            .into_iter()
            .map(|id| (id.start, id.end))
            .collect::<Vec<_>>();
        assert_eq!(retained_after, retained_before[layer]);
        let mut restored = PagedKeyValueCache::new(manager.clone(), layer, window).unwrap();
        assert_eq!(restored.offset(), 6);
        restored
            .update_for_attention(suffix.clone(), suffix.clone(), stream)
            .unwrap();
        let output = restored
            .paged_attention(
                &query,
                1.0,
                None,
                None,
                None,
                eredu_nn::AttentionArithmetic::Fused,
                stream,
            )
            .unwrap()
            .unwrap()
            .evaluated()
            .unwrap()
            .item::<f32>();
        assert!((output - uninterrupted[layer]).abs() < 1e-6);
    }
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn live_disk_budget_demotes_and_drop_removes_ephemeral_blocks() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let directory = tempfile::tempdir().unwrap();
    // MLX host-transfer buffers are page-aligned; permit one physical
    // allocation while keeping the logical device budget at two blocks.
    let options = PagedCacheOptions::new(2, 64, 32 * 1024, 1)
        .unwrap()
        .with_full_attention(true)
        .with_live_disk(directory.path(), 4096, 2)
        .unwrap();
    let manager = CacheResidencyManager::new(options).unwrap();
    let mut cache = PagedKeyValueCache::new(manager.clone(), 0, None).unwrap();
    let states = Array::from_slice(
        &[
            0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0,
        ],
        &[1, 1, 6, 2],
    );
    cache
        .update_and_fetch(states.clone(), states, stream)
        .unwrap();
    let mut report = manager.report().unwrap();
    for _ in 0..100 {
        if report.disk_demotions == 1 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
        report = manager.report().unwrap();
    }
    assert_eq!(report.disk_blocks, 1);
    assert_eq!(report.disk_demotions, 1);
    assert!(fs::read_dir(directory.path()).unwrap().next().is_some());
    drop(cache);
    drop(manager);
    assert!(fs::read_dir(directory.path()).unwrap().next().is_none());
}

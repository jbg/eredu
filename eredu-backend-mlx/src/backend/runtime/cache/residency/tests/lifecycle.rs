#[test]
fn layer_truncation_clears_only_the_selected_pages_and_mutable_tail() {
    let manager = CacheResidencyManager::new(
        PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let block_id = |global_layer| CacheBlockId {
        session_id: manager.session_id,
        global_layer,
        representation: CacheRepresentation::KeyValue,
        start: 0,
        end: 4,
        rank: None,
    };
    {
        let mut state = manager.lock().unwrap();
        for global_layer in 0..=1 {
            let id = block_id(global_layer);
            let mut location = missing_location(
                Path::new("/nonexistent/eredu-cache-test"),
                &format!("layer-{global_layer}.safetensors"),
            );
            location.persistent = true;
            insert_test_record(
                &mut state,
                CacheBlockRecord {
                    physical: MlxCacheBlockStorage::disk(id, location),
                    bytes: 0,
                    shapes: [vec![1, 1, 1, 1], vec![1, 1, 1, 1]],
                    dtypes: ["Float32".into(), "Float32".into()],
                    imported: false,
                },
                false,
                0,
            );
        }
    }
    manager.set_tail_state(0, 40, 5).unwrap();
    manager.set_tail_state(1, 56, 7).unwrap();
    let generation_before = manager.lock().unwrap().generation;

    manager
        .truncate_layer_transaction(1, CacheRepresentation::KeyValue, 0, None, 0)
        .unwrap();

    let state = manager.lock().unwrap();
    assert_eq!(state.generation, generation_before + 1);
    assert!(state.blocks.contains_key(&block_id(0)));
    assert!(!state.blocks.contains_key(&block_id(1)));
    assert_eq!(
        state.lifecycle.tail(0),
        Some(MutableCacheTail { bytes: 40, end: 5 })
    );
    assert_eq!(
        state.lifecycle.tail(1),
        Some(MutableCacheTail { bytes: 0, end: 0 })
    );
}

#[test]
fn process_pool_enforces_aggregate_device_budget_and_releases_membership() {
    let pool = CacheResidencyPool::new(CachePoolLimits::new(20, 20, 20, 0).unwrap());
    let options = PagedCacheOptions::new(1, 20, 20, 1)
        .unwrap()
        .with_pool(pool.clone())
        .unwrap();
    let first = CacheResidencyManager::new(options.clone()).unwrap();
    let second = CacheResidencyManager::new(options).unwrap();
    let first_layer_handle = first.clone();

    first.set_tail_state(0, 12, 1).unwrap();
    let error = second.set_tail_state(0, 12, 1).unwrap_err();
    assert!(matches!(
        error,
        CacheResidencyError::Pool(CachePoolError::BudgetExceeded {
            resource: CachePoolResource::Device,
            required: 24,
            budget: 20,
        })
    ));
    let report = pool.report().unwrap();
    assert_eq!(report.managers, 2);
    assert_eq!(report.current_device_bytes, 12);
    assert_eq!(report.peak_device_bytes, 12);

    drop(first);
    let report = pool.report().unwrap();
    assert_eq!(report.managers, 2);
    assert_eq!(report.current_device_bytes, 12);
    drop(first_layer_handle);
    let report = pool.report().unwrap();
    assert_eq!(report.managers, 1);
    assert_eq!(report.current_device_bytes, 0);
    drop(second);
    assert_eq!(pool.report().unwrap().managers, 0);
}

#[test]
fn per_cache_limits_cannot_exceed_their_process_pool() {
    let pool = CacheResidencyPool::new(CachePoolLimits::new(8, 4, 8, 0).unwrap());
    let error = PagedCacheOptions::new(1, 16, 4, 1)
        .unwrap()
        .with_pool(pool)
        .unwrap_err();
    assert!(matches!(
        error,
        CacheResidencyConfigurationError::InvalidOptions(_)
    ));
    assert!(error.to_string().contains("per-cache device budget 16"));
}

#[test]
fn disk_backed_device_blocks_bypass_a_zero_host_budget() {
    let directory = tempfile::tempdir().unwrap();
    let manager = CacheResidencyManager::new(PagedCacheOptions::new(1, 16, 0, 1).unwrap()).unwrap();
    let older = disk_test_id(0);
    let recent = disk_test_id(2);
    {
        let mut state = manager.lock().unwrap();
        for (id, disk) in [
            (
                older.clone(),
                Some(missing_location(directory.path(), "older.safetensors")),
            ),
            (recent.clone(), None),
        ] {
            insert_test_record(
                &mut state,
                CacheBlockRecord {
                    physical: MlxCacheBlockStorage::device(id.clone(), test_device_block(), disk),
                    bytes: 16,
                    shapes: [vec![1], vec![1]],
                    dtypes: ["Float32".into(), "Float32".into()],
                    imported: false,
                },
                false,
                0,
            );
        }
    }

    manager.rebalance(None, false).unwrap();
    let state = manager.lock().unwrap();
    assert_eq!(state.blocks.get(&older).unwrap().tier(), CacheTier::Disk);
    assert_eq!(state.blocks.get(&recent).unwrap().tier(), CacheTier::Device);
    assert_eq!(state.telemetry.report.current_host_bytes, 0);
    assert_eq!(state.telemetry.report.current_device_bytes, 16);
    assert_eq!(state.telemetry.report.current_disk_bytes, 16);
}

#[test]
fn per_layer_residency_report_is_bounded_and_losslessly_aggregated() {
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(1, u64::MAX, u64::MAX, 1).unwrap())
            .unwrap();
    let layer_count = CACHE_RESIDENCY_LAYER_REPORT_LIMIT + 3;
    {
        let mut state = manager.lock().unwrap();
        for global_layer in 0..layer_count {
            let representation = if global_layer % 2 == 0 {
                CacheRepresentation::KeyValue
            } else {
                CacheRepresentation::CompressedLatentRotary
            };
            let tier = match global_layer % 3 {
                0 => CacheTier::Device,
                1 => CacheTier::Host,
                _ => CacheTier::Disk,
            };
            let id = CacheBlockId {
                session_id: manager.session_id,
                global_layer,
                representation,
                start: 0,
                end: global_layer as i64 + 1,
                rank: None,
            };
            let physical = match tier {
                CacheTier::Device => {
                    MlxCacheBlockStorage::device(id.clone(), test_device_block(), None)
                }
                CacheTier::Host => MlxCacheBlockStorage::host(id.clone(), test_host_block(), None),
                CacheTier::Disk => MlxCacheBlockStorage::disk(
                    id.clone(),
                    missing_location(
                        Path::new("/tmp/eredu-mlx-cache-report-test"),
                        &format!("layer-{global_layer}.safetensors"),
                    ),
                ),
            };
            insert_test_record(
                &mut state,
                CacheBlockRecord {
                    physical,
                    bytes: global_layer as u64 + 1,
                    shapes: [vec![1], vec![1]],
                    dtypes: ["Float32".into(), "Float32".into()],
                    imported: false,
                },
                global_layer % 5 == 0,
                0,
            );
            state.lifecycle.set_tail(
                global_layer,
                MutableCacheTail {
                    bytes: 2,
                    end: global_layer as i64 + 1,
                },
            );
        }
    }

    let report = manager.report().unwrap();
    assert_eq!(report.per_layer.len(), CACHE_RESIDENCY_LAYER_REPORT_LIMIT);
    assert_eq!(report.per_layer_overflow_layers, 3);
    assert_eq!(
        report
            .per_layer
            .iter()
            .map(|layer| layer.global_layer)
            .collect::<Vec<_>>(),
        (0..CACHE_RESIDENCY_LAYER_REPORT_LIMIT).collect::<Vec<_>>()
    );

    let mut aggregate = CacheLayerResidencyStats::default();
    for layer in &report.per_layer {
        aggregate.accumulate(&layer.stats);
    }
    aggregate.accumulate(&report.per_layer_overflow);
    assert_eq!(aggregate.key_value_blocks, report.key_value_blocks);
    assert_eq!(
        aggregate.compressed_latent_blocks,
        report.compressed_latent_blocks
    );
    assert_eq!(aggregate.device_blocks, report.device_blocks);
    assert_eq!(aggregate.host_blocks, report.host_blocks);
    assert_eq!(aggregate.disk_blocks, report.disk_blocks);
    assert_eq!(aggregate.current_device_bytes, report.current_device_bytes);
    assert_eq!(aggregate.current_host_bytes, report.current_host_bytes);
    assert_eq!(aggregate.current_disk_bytes, report.current_disk_bytes);
    assert_eq!(aggregate.mutable_tail_bytes, report.mutable_tail_bytes);
    assert_eq!(
        aggregate.protected_recent_blocks,
        report.protected_recent_blocks
    );
    assert_eq!(
        aggregate.protected_prefix_blocks,
        report.protected_prefix_blocks
    );
    assert_eq!(report.logical_cached_tokens, layer_count as u64);
    assert_eq!(
        report.per_layer_overflow.logical_cached_tokens,
        ((CACHE_RESIDENCY_LAYER_REPORT_LIMIT + 1)..=layer_count)
            .map(|tokens| tokens as u64)
            .sum::<u64>()
    );
}

#[test]
fn per_layer_cumulative_attention_is_bounded_and_survives_clear() {
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(1, u64::MAX, u64::MAX, 1).unwrap())
            .unwrap();
    let layer_count = CACHE_RESIDENCY_LAYER_REPORT_LIMIT + 3;
    for global_layer in 0..layer_count {
        manager
            .record_attention_scan(
                global_layer,
                global_layer % 2 == 0,
                1,
                global_layer as u64 + 1,
                global_layer as u64 + 7,
            )
            .unwrap();
    }

    let report = manager.report().unwrap();
    assert_eq!(report.per_layer.len(), CACHE_RESIDENCY_LAYER_REPORT_LIMIT);
    assert_eq!(report.per_layer_overflow_layers, 0);
    assert_eq!(
        report
            .per_layer
            .iter()
            .map(|layer| layer.global_layer)
            .collect::<Vec<_>>(),
        (0..CACHE_RESIDENCY_LAYER_REPORT_LIMIT).collect::<Vec<_>>()
    );
    let mut aggregate = CacheLayerResidencyStats::default();
    for layer in &report.per_layer {
        aggregate.accumulate(&layer.stats);
    }
    aggregate.accumulate(&report.per_layer_overflow);
    assert_eq!(
        aggregate.prefill_full_attention_blocks,
        report.prefill_full_attention_blocks
    );
    assert_eq!(
        aggregate.prefill_full_attention_bytes,
        report.prefill_full_attention_bytes
    );
    assert_eq!(
        aggregate.decode_full_attention_blocks,
        report.decode_full_attention_blocks
    );
    assert_eq!(
        aggregate.decode_full_attention_bytes,
        report.decode_full_attention_bytes
    );
    assert_eq!(
        aggregate.attention_scratch_peak_bytes,
        report.attention_scratch_peak_bytes
    );
    assert_eq!(report.per_layer_overflow.prefill_full_attention_blocks, 2);
    assert_eq!(report.per_layer_overflow.decode_full_attention_blocks, 1);

    manager.clear().unwrap();
    let after_clear = manager.report().unwrap();
    assert_eq!(
        after_clear.per_layer.len(),
        CACHE_RESIDENCY_LAYER_REPORT_LIMIT
    );
    assert_eq!(
        after_clear.prefill_full_attention_blocks,
        report.prefill_full_attention_blocks
    );
    assert_eq!(
        after_clear.per_layer_overflow.decode_full_attention_bytes,
        report.per_layer_overflow.decode_full_attention_bytes
    );
    assert!(after_clear
        .per_layer
        .iter()
        .all(|layer| layer.stats.current_device_bytes == 0
            && layer.stats.current_host_bytes == 0
            && layer.stats.current_disk_bytes == 0));
}

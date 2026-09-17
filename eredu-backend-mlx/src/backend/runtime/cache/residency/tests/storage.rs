#[test]
fn retained_cache_storage_counts_all_tiers_aliases_and_buffered_shards_without_reads() {
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(1, 4096, 4096, 1).unwrap()).unwrap();
    let execution = cpu_stream();
    let root = Array::from_slice(&[0.5f32, -1.5], &[2]);
    let arrays = CacheBlockArrays::KeyValue {
        keys: root.clone(),
        values: root.clone(),
    };
    let host = HostCacheBlock::from_device_arrays(&arrays, &execution).unwrap();
    let host_bytes = host
        .buffers()
        .into_iter()
        .map(|buffer| buffer.capacity().unwrap() as u64)
        .sum::<u64>();
    let buffered: Arc<[u8]> = Arc::from([7u8, 3, 1, 9, 5]);
    let weak = Arc::downgrade(&buffered);
    let location = DiskLocation::ordinary(DiskLocationData {
        buffered: Some(buffered.clone()),
        ..missing_location_data(Path::new("missing-cold-cache-shard"), "never-opened")
    });
    let device_id = disk_test_id(0);
    let host_id = disk_test_id(1);
    {
        let mut state = manager.lock().unwrap();
        for (id, physical) in [
            (
                device_id.clone(),
                MlxCacheBlockStorage::device(device_id, arrays, Some(location.clone())),
            ),
            (
                host_id.clone(),
                MlxCacheBlockStorage::host(host_id, host, Some(location)),
            ),
        ] {
            insert_test_record(
                &mut state,
                CacheBlockRecord {
                    physical,
                    bytes: 16,
                    shapes: [vec![2], vec![2]],
                    dtypes: ["Float32".into(), "Float32".into()],
                    imported: false,
                    original_discard: None,
                    _metadata_funding: None,
                },
                false,
                0,
            );
            assert!(state.blocks.contains_key(&id));
        }
    }
    let expected = root.allocation_info().unwrap().unwrap().bytes() as u64
        + host_bytes
        + buffered.len() as u64;
    let mut storage = crate::backend::runtime::cache::state::retained_state_storage(
        [&root],
        [&manager, &manager],
    )
    .unwrap();
    assert_eq!(storage.byte_bound().unwrap(), Some(expected));
    storage.merge(manager.retained_storage().unwrap()).unwrap();
    assert_eq!(storage.byte_bound().unwrap(), Some(expected));
    assert_eq!(manager.lock().unwrap().blocks.len(), 2);
    drop((manager, root, buffered));
    assert_eq!(weak.upgrade().unwrap().as_ref(), &[7, 3, 1, 9, 5]);
    assert_eq!(storage.byte_bound().unwrap(), Some(expected));
    drop(storage);
    assert!(weak.upgrade().is_none());
}

#[test]
fn retained_cache_storage_never_reaps_pending_host_completion() {
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(1, 4096, 4096, 1).unwrap()).unwrap();
    let completion = Arc::new(HostDemotionCompletion::default());
    completion.finish(Ok(test_host_block()));
    let id = disk_test_id(0);
    let ticket = HostDemotionTicket {
        operation_id: 81,
        id: id.clone(),
        reserved_host_bytes: 8,
        completion: completion.clone(),
    };
    insert_test_record(
        &mut manager.lock().unwrap(),
        CacheBlockRecord {
            physical: test_demoting(test_device_block(), ticket),
            bytes: 8,
            shapes: [vec![1], vec![1]],
            dtypes: ["Float32".into(), "Float32".into()],
            imported: false,
            original_discard: None,
            _metadata_funding: None,
        },
        false,
        0,
    );
    let phase = manager.lock().unwrap().blocks[&id].physical.phase();
    assert_eq!(
        manager.retained_storage().unwrap().byte_bound().unwrap(),
        None
    );
    assert_eq!(manager.lock().unwrap().blocks[&id].physical.phase(), phase);
    assert!(completion.result.lock().unwrap().as_ref().unwrap().is_ok());
}

#[test]
fn retained_cache_storage_cannot_hide_worker_payloads_outside_the_block_map() {
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(1, 4096, 4096, 1).unwrap()).unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    assert!(
        manager
            .inner
            .host_demotion_worker
            .sender
            .send(super::HostDemotionRequest::Pause {
                started: started_tx,
                release: release_rx,
                retained: Arc::new(()),
                finished: finished_tx,
            })
            .is_ok()
    );
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(manager.lock().unwrap().blocks.is_empty());
    let retained = manager.retained_storage().unwrap();
    let unknown = retained.byte_bound().unwrap();
    release_tx.send(()).unwrap();
    finished_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(unknown, None);
    let deadline = std::time::Instant::now() + Duration::from_secs(1);
    while manager
        .inner
        .host_demotion_worker
        .active_payload
        .load(std::sync::atomic::Ordering::Acquire)
    {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(
        manager.retained_storage().unwrap().byte_bound().unwrap(),
        Some(0)
    );
    assert_eq!(
        retained.byte_bound().unwrap(),
        None,
        "old unknown inventory cannot become certified later"
    );
}

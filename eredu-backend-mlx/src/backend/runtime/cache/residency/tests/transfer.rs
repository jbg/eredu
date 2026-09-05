#[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
#[test]
fn cpu_cache_backend_preserves_ordering_persistence_and_aggregate_budgets() {
    exercise_backend_cache_lifecycle(
        Device::new(DeviceType::Cpu, 0),
        HostTransferStorageKind::Cpu,
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "explicit Metal cache-residency test; run outside the sandbox"]
fn metal_cache_backend_preserves_ordering_persistence_and_aggregate_budgets() {
    exercise_backend_cache_lifecycle(
        Device::new(DeviceType::Gpu, 0),
        HostTransferStorageKind::MetalShared,
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "explicit CUDA cache-residency test; requires a CUDA-capable host"]
fn cuda_cache_backend_preserves_ordering_persistence_and_aggregate_budgets() {
    assert!(safemlx::cuda::is_available().unwrap());
    exercise_backend_cache_lifecycle(
        Device::new(DeviceType::Gpu, 0),
        HostTransferStorageKind::CudaPinned,
    );
}

#[test]
fn two_block_prefetch_uses_a_dedicated_cpu_stream_and_bounds_leases() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let host_capacity = two_buffer_host_capacity(2 * size_of::<f32>());
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(2, 32, host_capacity * 2, 1).unwrap())
            .unwrap();
    let ids = (0..3)
        .map(|index| {
            manager
                .seal_block(
                    0,
                    index * 2,
                    index * 2 + 2,
                    None,
                    execution_key_value_block(&stream),
                    false,
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    let ids = vec![ids[0].clone(), ids[2].clone(), ids[1].clone()];

    let mut blocks = manager.prefetch_blocks(ids, &stream).unwrap();
    let (execution_index, transfer_index) = blocks.stream_indices().unwrap();
    assert_ne!(execution_index, transfer_index);

    let first = blocks.next_block().unwrap().unwrap();
    assert_eq!(blocks.pending_len(), 1);
    let consumed = first.arrays().arrays()[0].square(&stream).unwrap();
    let consumed = async_eval_with_event([&consumed]).unwrap();
    drop(first);

    let second = blocks.next_block().unwrap().unwrap();
    assert_eq!(blocks.pending_len(), 1);
    drop(second);
    let third = blocks.next_block().unwrap().unwrap();
    assert_eq!(blocks.pending_len(), 0);
    drop(third);
    assert!(blocks.next_block().unwrap().is_none());

    // The submitted consumer and queued waits retain their cache arrays
    // after every public lease has been released.
    consumed.synchronize().unwrap();
}

#[test]
fn two_block_prefetch_falls_back_under_a_one_block_device_budget() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let host_capacity = two_buffer_host_capacity(2 * size_of::<f32>());
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(2, 16, host_capacity * 2, 1).unwrap())
            .unwrap();
    let ids = (0..3)
        .map(|index| {
            manager
                .seal_block(
                    0,
                    index * 2,
                    index * 2 + 2,
                    None,
                    execution_key_value_block(&stream),
                    false,
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut blocks = manager.prefetch_blocks(ids, &stream).unwrap();
    for _ in 0..3 {
        let lease = blocks.next_block().unwrap().unwrap();
        assert_eq!(blocks.pending_len(), 0);
        drop(lease);
    }
    assert!(blocks.next_block().unwrap().is_none());
    let report = manager.report().unwrap();
    assert!(report.current_device_bytes <= 16);
    assert_eq!(report.failures, 0);
}

#[test]
#[ignore = "explicit Metal paged-cache prefetch test; run on a Metal host"]
fn two_block_metal_prefetch_is_gpu_ordered_without_host_synchronization() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let block_bytes = 2 * 1024 * 1024 * std::mem::size_of::<f32>() as u64;
    let manager = CacheResidencyManager::new(
        PagedCacheOptions::new(1024, block_bytes * 3, block_bytes, 1).unwrap(),
    )
    .unwrap();
    let ids = (0..3)
        .map(|index| {
            let keys = Array::ones::<f32>(&[1024, 1024], &stream).unwrap();
            let values = Array::ones::<f32>(&[1024, 1024], &stream).unwrap();
            manager
                .seal_block(
                    0,
                    index * 1024,
                    index * 1024 + 1024,
                    None,
                    CacheBlockArrays::KeyValue { keys, values },
                    false,
                )
                .unwrap()
        })
        .collect::<Vec<_>>();
    // Stage one exact host block without creating device-budget pressure;
    // the promotion assertion below then observes only the async copy.
    let host_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let device_arrays = manager
        .lock()
        .unwrap()
        .blocks
        .get(&ids[0])
        .unwrap()
        .device_arrays()
        .unwrap()
        .clone();
    let host_block = HostCacheBlock::from_device_arrays(&device_arrays, &host_stream).unwrap();
    {
        let mut state = manager.lock().unwrap();
        let record = state.blocks.get_mut(&ids[0]).unwrap();
        record.physical = MlxCacheBlockStorage::host(ids[0].clone(), host_block, None);
        super::update_report_totals(&mut state);
    }
    let ids = vec![ids[0].clone(), ids[2].clone(), ids[1].clone()];

    let transfer = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let blocker_lhs = Array::ones::<f32>(&[4096, 4096], &transfer).unwrap();
    let blocker_rhs = Array::ones::<f32>(&[4096, 4096], &transfer).unwrap();
    let blocker = blocker_lhs.matmul(&blocker_rhs, &transfer).unwrap();
    safemlx::transforms::async_eval([&blocker]).unwrap();
    let direct = manager.prepare_block_transfer(&ids[0], &transfer).unwrap();
    assert!(
        direct
            .completions
            .iter()
            .any(|completion| !completion.is_complete().unwrap()),
        "paged-cache promotion blocked the host"
    );
    direct.wait_on(&stream).unwrap();
    let consumed = direct.arrays().arrays()[0].square(&stream).unwrap();
    let completion = async_eval_with_event([&consumed]).unwrap();
    drop(direct);
    completion.synchronize().unwrap();

    let mut blocks = manager.prefetch_blocks(ids, &stream).unwrap();
    let (execution_index, transfer_index) = blocks.stream_indices().unwrap();
    assert_ne!(execution_index, transfer_index);
    let first = blocks.next_block().unwrap().unwrap();
    assert_eq!(blocks.pending_len(), 1);
    drop(first);
    drop(blocks);
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn promotion_waits_for_pending_write_without_overcommitting_host_storage() {
    let directory = tempfile::tempdir().unwrap();
    let host_capacity = two_buffer_host_capacity(2 * size_of::<f32>());
    let options = PagedCacheOptions::new(2, 32, host_capacity, 1)
        .unwrap()
        .with_live_disk(directory.path(), 1024, 1)
        .unwrap();
    let manager = CacheResidencyManager::new(options).unwrap();
    let worker = manager.inner.disk_worker.as_ref().unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let blocker = worker
        .prepare(
            CacheIoOperationKey {
                generation: 0,
                id: disk_test_id(99),
                kind: CacheIoOperationKind::Read,
            },
            DiskTask::Pause {
                started: started_tx,
                release: release_rx,
            },
        )
        .unwrap();
    let blocker_ticket = blocker.ticket.clone();
    blocker.enqueue().unwrap();
    started_rx.recv().unwrap();

    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let first = manager
        .seal_block(0, 0, 2, None, execution_key_value_block(&stream), false)
        .unwrap();
    manager
        .seal_block(0, 2, 4, None, execution_key_value_block(&stream), false)
        .unwrap();
    manager
        .seal_block(0, 4, 6, None, execution_key_value_block(&stream), false)
        .unwrap();
    let report = manager.report().unwrap();
    assert_eq!(report.current_host_bytes, host_capacity);
    assert_eq!(report.in_flight_write_bytes, host_capacity);

    let promotion_manager = manager.clone();
    let (promoted_tx, promoted_rx) = mpsc::channel();
    let promotion_thread = thread::spawn(move || {
        let promotion_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let result = promotion_manager.lease_block(&first, &promotion_stream);
        promoted_tx.send(result.map(drop)).unwrap();
    });
    match promoted_rx.recv_timeout(Duration::from_millis(20)) {
        Err(mpsc::RecvTimeoutError::Timeout) => {}
        result => panic!("promotion completed before host capacity was released: {result:?}"),
    }
    let report = manager.report().unwrap();
    assert_eq!(report.current_host_bytes, host_capacity);
    assert_eq!(report.in_flight_write_bytes, host_capacity);
    assert!(report.current_host_bytes <= manager.options().host_budget_bytes());

    release_tx.send(()).unwrap();
    assert!(matches!(blocker_ticket.wait().unwrap(), DiskResult::Test));
    promoted_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("promotion did not finish after writeback released host capacity")
        .unwrap();
    promotion_thread.join().unwrap();
    let report = manager.report().unwrap();
    assert!(report.current_host_bytes <= manager.options().host_budget_bytes());
    assert!(report.peak_host_bytes <= manager.options().host_budget_bytes());
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn host_to_disk_demotion_returns_before_background_write_completes() {
    let directory = tempfile::tempdir().unwrap();
    let host_capacity = two_buffer_host_capacity(2 * size_of::<f32>());
    let options = PagedCacheOptions::new(2, 16, host_capacity, 1)
        .unwrap()
        .with_live_disk(directory.path(), 1024, 1)
        .unwrap();
    let manager = CacheResidencyManager::new(options).unwrap();
    let worker = manager.inner.disk_worker.as_ref().unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let blocker = worker
        .prepare(
            CacheIoOperationKey {
                generation: 0,
                id: disk_test_id(99),
                kind: CacheIoOperationKind::Read,
            },
            DiskTask::Pause {
                started: started_tx,
                release: release_rx,
            },
        )
        .unwrap();
    let blocker_ticket = blocker.ticket.clone();
    blocker.enqueue().unwrap();
    started_rx.recv().unwrap();

    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    manager
        .seal_block(0, 0, 2, None, execution_key_value_block(&stream), false)
        .unwrap();
    let second = execution_key_value_block(&stream);
    let background_manager = manager.clone();
    let (sealed_tx, sealed_rx) = mpsc::channel();
    let seal_thread = thread::spawn(move || {
        sealed_tx
            .send(background_manager.seal_block(0, 2, 4, None, second, false))
            .unwrap();
    });

    sealed_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("demotion waited for the blocked disk worker")
        .unwrap();
    let report = manager.report().unwrap();
    assert_eq!(report.in_flight_write_blocks, 1);
    assert_eq!(report.in_flight_write_bytes, host_capacity);
    assert_eq!(report.current_host_bytes, host_capacity);
    assert_eq!(report.host_blocks, 1);
    assert!(report.current_host_bytes <= manager.options().host_budget_bytes());
    assert_eq!(report.disk_demotions, 0);

    // A third block needs the same host slot. It must wait for the pending
    // write to commit instead of retaining another host allocation beyond
    // the byte budget.
    let third = execution_key_value_block(&stream);
    let waiting_manager = manager.clone();
    let (third_tx, third_rx) = mpsc::channel();
    let third_thread = thread::spawn(move || {
        third_tx
            .send(waiting_manager.seal_block(0, 4, 6, None, third, false))
            .unwrap();
    });
    assert!(matches!(
        third_rx.recv_timeout(Duration::from_millis(20)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    let report = manager.report().unwrap();
    assert_eq!(report.current_host_bytes, host_capacity);
    assert_eq!(report.in_flight_write_bytes, host_capacity);

    release_tx.send(()).unwrap();
    assert!(matches!(blocker_ticket.wait().unwrap(), DiskResult::Test));
    seal_thread.join().unwrap();
    third_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("host-capacity wait did not finish after writeback")
        .unwrap();
    third_thread.join().unwrap();
    for _ in 0..100 {
        let report = manager.report().unwrap();
        if report.disk_demotions >= 2 {
            assert_eq!(report.in_flight_write_blocks, 0);
            assert_eq!(report.disk_blocks, 2);
            assert!(report.current_host_bytes <= manager.options().host_budget_bytes());
            assert!(report.peak_host_bytes <= manager.options().host_budget_bytes());
            assert!(report.in_flight_waits >= 1);
            let layer = report
                .per_layer
                .iter()
                .find(|layer| layer.global_layer == 0)
                .unwrap();
            assert_eq!(layer.stats.disk_demotions, report.disk_demotions);
            assert_eq!(layer.stats.in_flight_waits, report.in_flight_waits);
            return;
        }
        thread::sleep(Duration::from_millis(10));
    }
    panic!("background disk write did not commit");
}

#[test]
fn failed_asynchronous_host_demotion_restores_the_device_state() {
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(1, 16, 16, 1).unwrap()).unwrap();
    let id = disk_test_id(0);
    let completion = Arc::new(HostDemotionCompletion::default());
    completion.finish(Err(CacheResidencyError::Runtime(
        "injected host demotion failure".into(),
    )));
    let ticket = HostDemotionTicket {
        operation_id: 71,
        id: id.clone(),
        reserved_host_bytes: 8,
        completion,
    };
    insert_test_record(
        &mut manager.lock().unwrap(),
        CacheBlockRecord {
            physical: test_demoting(test_device_block(), ticket.clone()),
            bytes: 8,
            shapes: [vec![1], vec![1]],
            dtypes: ["Float32".into(), "Float32".into()],
            imported: false,
        },
        false,
        0,
    );

    let error = manager.finish_device_demotion(&ticket).unwrap_err();
    assert!(error.to_string().contains("injected host demotion failure"));
    let state = manager.lock().unwrap();
    assert_eq!(
        state.blocks.get(&id).unwrap().physical.phase(),
        CacheStoragePhase::Device
    );
    drop(state);
    let report = manager.report().unwrap();
    assert_eq!(report.current_device_bytes, 8);
    assert_eq!(report.current_host_bytes, 0);
    assert_eq!(report.in_flight_host_demotion_blocks, 0);
    assert_eq!(report.failures, 1);
}

#[test]
fn retiring_host_demotion_remains_charged_during_generation_reset() {
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(1, 16, 16, 1).unwrap()).unwrap();
    let id = disk_test_id(0);
    let completion = Arc::new(HostDemotionCompletion::default());
    let ticket = HostDemotionTicket {
        operation_id: 72,
        id: id.clone(),
        reserved_host_bytes: 8,
        completion: Arc::clone(&completion),
    };
    insert_test_record(
        &mut manager.lock().unwrap(),
        CacheBlockRecord {
            physical: test_demoting(test_device_block(), ticket),
            bytes: 8,
            shapes: [vec![1], vec![1]],
            dtypes: ["Float32".into(), "Float32".into()],
            imported: false,
        },
        false,
        0,
    );

    let clearing_manager = manager.clone();
    let clearing = thread::spawn(move || clearing_manager.clear());
    let deadline = std::time::Instant::now() + Duration::from_secs(1);
    loop {
        let report = manager.report().unwrap();
        if report.in_flight_host_demotion_blocks == 1 && manager.lock().unwrap().blocks.is_empty() {
            assert_eq!(report.current_device_bytes, 8);
            assert_eq!(report.current_host_bytes, 8);
            assert_eq!(report.in_flight_host_demotion_bytes, 8);
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "clear did not publish its retiring demotion reservation"
        );
        thread::yield_now();
    }

    completion.finish(Err(CacheResidencyError::Runtime(
        "discarded generation transfer failed".into(),
    )));
    clearing.join().unwrap().unwrap();
    let report = manager.report().unwrap();
    assert_eq!(report.current_device_bytes, 0);
    assert_eq!(report.current_host_bytes, 0);
    assert_eq!(report.in_flight_host_demotion_blocks, 0);
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn asynchronous_host_demotion_charges_both_allocations_until_reconciled() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let host_capacity = two_buffer_host_capacity(2 * size_of::<f32>());
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(2, 32, host_capacity, 1).unwrap())
            .unwrap();
    manager.bind_transfer_device(&stream).unwrap();
    let arrays = execution_key_value_block(&stream);
    let device_pointers = f32_storage_pointers(&arrays);
    let id = manager.seal_block(0, 0, 2, None, arrays, false).unwrap();

    let ticket = manager.begin_device_demotion(&id).unwrap();
    {
        let state = manager.lock().unwrap();
        let record = state.blocks.get(&id).unwrap();
        assert_eq!(record.physical.phase(), CacheStoragePhase::DemotingToHost);
        let arrays = record.physical.device_resource().unwrap();
        let live = record.physical.host_demotion().unwrap();
        assert_eq!(live.operation_id, ticket.operation_id);
        assert_eq!(f32_storage_pointers(arrays), device_pointers);
    }
    let report = manager.report().unwrap();
    assert_eq!(report.device_blocks, 1);
    assert_eq!(report.host_blocks, 1);
    assert_eq!(report.current_device_bytes, 16);
    assert_eq!(report.current_host_bytes, host_capacity);
    assert_eq!(report.in_flight_host_demotion_blocks, 1);
    assert_eq!(report.in_flight_host_demotion_bytes, host_capacity);
    assert_eq!(report.peak_in_flight_host_demotion_bytes, host_capacity);
    let layer = report
        .per_layer
        .iter()
        .find(|layer| layer.global_layer == 0)
        .unwrap();
    assert_eq!(layer.stats.in_flight_host_demotion_blocks, 1);
    assert_eq!(layer.stats.in_flight_host_demotion_bytes, host_capacity);

    manager.finish_device_demotion(&ticket).unwrap();
    let report = manager.report().unwrap();
    assert_eq!(report.device_blocks, 0);
    assert_eq!(report.host_blocks, 1);
    assert_eq!(report.current_device_bytes, 0);
    assert_eq!(report.current_host_bytes, host_capacity);
    assert_eq!(report.in_flight_host_demotion_blocks, 0);
    assert_eq!(report.in_flight_host_demotion_bytes, 0);
    assert_eq!(report.host_demotions, 1);
    assert_eq!(report.transfer_bytes, 16);
}

#[test]
fn cache_pool_charges_physical_host_allocation_capacity() {
    let stream = cpu_stream();
    let arrays = CacheBlockArrays::KeyValue {
        keys: Array::zeros::<f32>(&[1, 1, 1, 5000], &stream).unwrap(),
        values: Array::zeros::<f32>(&[1, 1, 1, 5000], &stream).unwrap(),
    };
    let logical = arrays.bytes();
    let capacity = host_cache_capacity_upper_bound(&arrays).unwrap();
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(1, logical, capacity, 1).unwrap())
            .unwrap();
    manager.bind_transfer_device(&stream).unwrap();
    let id = manager.seal_block(0, 0, 1, None, arrays, false).unwrap();
    let ticket = manager.begin_device_demotion(&id).unwrap();
    manager.finish_device_demotion(&ticket).unwrap();

    let report = manager.report().unwrap();
    assert_eq!(report.current_host_bytes, capacity);
    assert_eq!(report.peak_host_bytes, capacity);
    assert_eq!(
        manager.pool().report().unwrap().current_host_bytes,
        capacity
    );
    let state = manager.lock().unwrap();
    let block = state.blocks.get(&id).unwrap().host_block().unwrap();
    assert_eq!(block.bytes().unwrap(), logical);
    assert_eq!(block.capacity().unwrap(), capacity);
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn clear_waits_for_asynchronous_host_demotion_resources() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let host_capacity = two_buffer_host_capacity(2 * size_of::<f32>());
    let manager =
        CacheResidencyManager::new(PagedCacheOptions::new(2, 32, host_capacity, 1).unwrap())
            .unwrap();
    manager.bind_transfer_device(&stream).unwrap();
    let id = manager
        .seal_block(0, 0, 2, None, execution_key_value_block(&stream), false)
        .unwrap();
    let ticket = manager.begin_device_demotion(&id).unwrap();

    manager.clear().unwrap();
    assert!(ticket.wait().is_ok());
    let report = manager.report().unwrap();
    assert_eq!(report.device_blocks, 0);
    assert_eq!(report.host_blocks, 0);
    assert_eq!(report.current_device_bytes, 0);
    assert_eq!(report.current_host_bytes, 0);
    assert_eq!(report.in_flight_host_demotion_blocks, 0);
}

#[test]
#[ignore = "requires MLX runtime execution"]
fn host_demotion_uses_typed_buffers_and_promotion_rebuilds_device_arrays() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    // Two 16-byte blocks fit on the device. A third block forces the oldest
    // one into backend-selected host-transfer storage while retaining one
    // recent block.
    let host_capacity = two_buffer_host_capacity(2 * size_of::<f32>());
    let options = PagedCacheOptions::new(2, 32, host_capacity, 1).unwrap();
    let manager = CacheResidencyManager::new(options).unwrap();
    manager.bind_transfer_device(&stream).unwrap();
    let first_arrays = execution_key_value_block(&stream);
    let first = manager
        .seal_block(0, 0, 2, None, first_arrays, false)
        .unwrap();
    manager
        .seal_block(0, 2, 4, None, execution_key_value_block(&stream), false)
        .unwrap();
    manager
        .seal_block(0, 4, 6, None, execution_key_value_block(&stream), false)
        .unwrap();

    let first_host_capacity = {
        let state = manager.lock().unwrap();
        let record = state.blocks.get(&first).unwrap();
        assert_eq!(record.tier(), CacheTier::Host);
        let block = record.host_block().unwrap();
        let [first, second] = block.buffers();
        first.capacity().unwrap() + second.capacity().unwrap()
    };
    assert!(first_host_capacity >= 16);
    let report = manager.report().unwrap();
    assert_eq!(report.device_blocks, 2);
    assert_eq!(report.host_blocks, 1);
    assert_eq!(report.current_device_bytes, 32);
    assert_eq!(report.current_host_bytes, host_capacity);

    let lease = manager.lease_block(&first, &stream).unwrap();
    let promoted_pointers = f32_storage_pointers(lease.arrays());
    // A demoted allocation may reuse the virtual address of its released
    // source on unified-memory systems, so pointer inequality is not a
    // storage-identity invariant. The typed buffer capacity is verified
    // above; promotion is verified by values and the live device record.
    match lease.arrays() {
        CacheBlockArrays::KeyValue { keys, values } => {
            assert_eq!(keys.evaluated().unwrap().as_slice::<f32>(), &[0.0, 0.0]);
            assert_eq!(values.evaluated().unwrap().as_slice::<f32>(), &[1.0, 1.0]);
        }
        CacheBlockArrays::CompressedLatentRotary { .. } => unreachable!(),
    }
    {
        let state = manager.lock().unwrap();
        let record = state.blocks.get(&first).unwrap();
        assert_eq!(record.tier(), CacheTier::Device);
        assert_eq!(
            f32_storage_pointers(record.device_arrays().unwrap()),
            promoted_pointers
        );
    }
    drop(lease);

    let report = manager.report().unwrap();
    assert_eq!(report.device_blocks, 2);
    assert_eq!(report.host_blocks, 1);
    assert_eq!(report.current_device_bytes, 32);
    assert_eq!(report.current_host_bytes, host_capacity);
    assert_eq!(report.host_promotions, 1);
    assert_eq!(report.host_demotions, 2);
    let layer = report
        .per_layer
        .iter()
        .find(|layer| layer.global_layer == 0)
        .unwrap();
    assert_eq!(layer.stats.host_promotions, report.host_promotions);
    assert_eq!(layer.stats.host_demotions, report.host_demotions);
    assert_eq!(layer.stats.transfer_bytes, report.transfer_bytes);
    assert_eq!(layer.stats.demand_misses, report.demand_misses);
    assert_eq!(first.representation, CacheRepresentation::KeyValue);
}

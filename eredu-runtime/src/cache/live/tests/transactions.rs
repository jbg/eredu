fn test_host_writing(block: TestHostPage, ticket: TestDiskTicket) -> TestStorage {
    let mut physical = TestStorage::host(ticket.key.id.clone(), block, None);
    physical
        .begin_write(CacheIoEntry {
            ticket,
            reserved_host_bytes: None,
        })
        .unwrap();
    physical
}

fn test_demoting(arrays: TestPage, ticket: TestDemotionTicket) -> TestStorage {
    let mut physical = TestStorage::device(ticket.id.clone(), arrays, None);
    physical.begin_host_demotion(ticket).unwrap();
    physical
}

fn insert_test_record(
    state: &mut TestState,
    record: TestRecord,
    protected_prefix: bool,
    leases: usize,
) {
    let id = record.physical.id().clone();
    state
        .lifecycle
        .insert(id.clone(), protected_prefix)
        .unwrap();
    for _ in 0..leases {
        state.lifecycle.acquire(&id).unwrap();
    }
    assert!(state.blocks.insert(id, record).is_none());
}


#[test]
fn manager_drop_does_not_lock_worker_retained_state_and_final_owner_cleans_files() {
    let directory = tempfile::tempdir().unwrap();
    let location = missing_location(directory.path(), "ephemeral.safetensors");
    fs::write(&location.path, b"owned until the final state retires").unwrap();
    let path = location.path.clone();
    let manager =
        Cache::new(PagedCacheOptions::new(1, 64, 64, 1).unwrap()).unwrap();
    {
        let mut state = manager.lock().unwrap();
        insert_test_record(
            &mut state,
            TestRecord {
                physical: TestStorage::disk(disk_test_id(0), location),
                bytes: 0,
                shapes: [vec![1], vec![1]],
                dtypes: ["Float32".into(), "Float32".into()],
                imported: false,
            },
            false,
            0,
        );
    }
    let retained_state = Arc::clone(&manager.inner.state);
    let (locked_tx, locked_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    thread::spawn(move || {
        let state = retained_state.lock().unwrap();
        let _ = locked_tx.send(());
        let _ = release_rx.recv();
        drop(state);
        drop(retained_state);
        let _ = finished_tx.send(());
    });
    locked_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    let (dropped_tx, dropped_rx) = mpsc::channel();
    thread::spawn(move || {
        drop(manager);
        let _ = dropped_tx.send(());
    });
    let dropped_before_release = dropped_rx.recv_timeout(Duration::from_secs(1));
    let file_still_owned = path.exists();
    release_tx.send(()).unwrap();
    dropped_before_release.unwrap();
    assert!(file_still_owned);
    finished_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(!path.exists());
}

#[test]
fn process_pool_enforces_aggregate_device_budget_and_releases_membership() {
    let pool = CacheResidencyPool::new(CachePoolLimits::new(20, 20, 20, 0).unwrap());
    let options = PagedCacheOptions::new(1, 20, 20, 1)
        .unwrap()
        .with_pool(pool.clone())
        .unwrap();
    let first = Cache::new(options.clone()).unwrap();
    let second = Cache::new(options).unwrap();
    let first_layer_handle = first.clone();

    first.set_tail_state(0, CacheRepresentation::KeyValue, 12, 1).unwrap();
    let error = second.set_tail_state(0, CacheRepresentation::KeyValue, 12, 1).unwrap_err();
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
fn disk_backed_device_blocks_bypass_a_zero_host_budget() {
    let directory = tempfile::tempdir().unwrap();
    let manager = Cache::new(PagedCacheOptions::new(1, 16, 0, 1).unwrap()).unwrap();
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
                TestRecord {
                    physical: TestStorage::device(id.clone(), test_device_block(), disk),
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
fn promoted_and_cancelled_writes_retain_host_reservations_until_release() {
    let directory = tempfile::tempdir().unwrap();
    let host_block = test_host_block();
    let host_capacity = host_block.capacity().unwrap();
    let pool = CacheResidencyPool::new(
        CachePoolLimits::new(16, host_capacity, host_capacity, 1024).unwrap(),
    );
    let options = PagedCacheOptions::new(1, 16, host_capacity, 1)
        .unwrap()
        .with_live_disk(directory.path(), 1024, 1)
        .unwrap()
        .with_pool(pool.clone())
        .unwrap();
    let manager = Cache::new(options).unwrap();
    let worker = manager.inner.disk_worker.as_ref().unwrap();
    let id = disk_test_id(0);
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let key = CacheIoOperationKey {
        generation: 0,
        id: id.clone(),
        kind: CacheIoOperationKind::Write,
    };
    let staged = stage_live_cache_block(
        directory.path(), &key, super::host_cache_tensors(&host_block).unwrap(),
    ).unwrap();
    let submission = worker
        .prepare(
            key.clone(),
            DiskTask::PauseWrite {
                started: started_tx,
                release: release_rx,
                staged,
                commit: DiskWriteCommit {
                    state: Arc::downgrade(&manager.inner.state),
                    key: key.clone(),
                    reservation_id: 7,
                    armed: true,
                },
            },
        )
        .unwrap();
    let ticket = submission.ticket.clone();
    submission.enqueue().unwrap();
    started_rx.recv().unwrap();
    {
        let mut state = manager.lock().unwrap();
        state.host_write_reservations.insert(
            key,
            HostWriteReservation {
                reservation_id: 7,
                logical_bytes: 16,
                host_capacity,
                ticket: ticket.clone(),
            },
        );
        insert_test_record(
            &mut state,
            TestRecord {
                physical: test_host_writing(host_block, ticket.clone()),
                bytes: 16,
                shapes: [vec![1], vec![1]],
                dtypes: ["Float32".into(), "Float32".into()],
                imported: false,
            },
            false,
            0,
        );
    }

    let report = manager.report().unwrap();
    assert_eq!(report.current_host_bytes, host_capacity);
    assert_eq!(report.in_flight_write_bytes, host_capacity);
    assert_eq!(report.host_blocks, 1);
    let aggregate = pool.report().unwrap();
    assert_eq!(aggregate.current_host_bytes, host_capacity);
    assert_eq!(aggregate.current_transfer_in_flight_bytes, host_capacity);
    assert_eq!(aggregate.current_disk_bytes, 16);

    // A pending host write is encoded only by the Host variant. Promotion
    // waits for that write rather than creating an incoherent device block
    // with an attached host-write operation.
    assert_eq!(
        manager
            .lock()
            .unwrap()
            .blocks
            .get(&id)
            .unwrap()
            .physical
            .phase(),
        CacheStoragePhase::HostWriting
    );

    let clear_manager = manager.clone();
    let (cleared_tx, cleared_rx) = mpsc::channel();
    let clear_thread = thread::spawn(move || {
        cleared_tx.send(clear_manager.clear()).unwrap();
    });
    assert!(matches!(
        ticket.wait(),
        Err(CacheResidencyError::DiskOperationCancelled { generation: 0 })
    ));
    assert!(matches!(
        cleared_rx.recv_timeout(Duration::from_millis(20)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    let report = manager.report().unwrap();
    assert_eq!(report.cancellations, 1);
    assert_eq!(report.host_blocks, 0);
    assert_eq!(report.current_host_bytes, host_capacity);
    assert_eq!(report.in_flight_write_bytes, host_capacity);
    let aggregate = pool.report().unwrap();
    assert_eq!(aggregate.current_host_bytes, host_capacity);
    assert_eq!(aggregate.current_transfer_in_flight_bytes, host_capacity);
    assert_eq!(aggregate.current_disk_bytes, 16);
    release_tx.send(()).unwrap();
    cleared_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("clear did not finish after the write released its arrays")
        .unwrap();
    clear_thread.join().unwrap();
    let report = manager.report().unwrap();
    assert_eq!(report.current_host_bytes, 0);
    assert_eq!(report.in_flight_write_bytes, 0);
    let aggregate = pool.report().unwrap();
    assert_eq!(aggregate.current_host_bytes, 0);
    assert_eq!(aggregate.current_transfer_in_flight_bytes, 0);
    assert_eq!(aggregate.current_disk_bytes, 0);
    assert!(directory.path().read_dir().unwrap().next().is_none());
}

#[test]
fn failed_asynchronous_host_demotion_restores_the_device_state() {
    let manager =
        Cache::new(PagedCacheOptions::new(1, 16, 16, 1).unwrap()).unwrap();
    let id = disk_test_id(0);
    let completion = Arc::new(pending_demotion());
    completion.finish(Err(CacheResidencyError::Runtime(
        "injected host demotion failure".into(),
    )));
    let ticket = TestDemotionTicket {
        operation_id: 71,
        id: id.clone(),
        reserved_host_bytes: 8,
        completion,
    };
    insert_test_record(
        &mut manager.lock().unwrap(),
        TestRecord {
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
        Cache::new(PagedCacheOptions::new(1, 16, 16, 1).unwrap()).unwrap();
    let id = disk_test_id(0);
    let completion = Arc::new(pending_demotion());
    let ticket = TestDemotionTicket {
        operation_id: 72,
        id: id.clone(),
        reserved_host_bytes: 8,
        completion: Arc::clone(&completion),
    };
    insert_test_record(
        &mut manager.lock().unwrap(),
        TestRecord {
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

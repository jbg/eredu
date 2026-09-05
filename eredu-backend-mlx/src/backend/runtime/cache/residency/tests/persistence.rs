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
    let manager = CacheResidencyManager::new(options).unwrap();
    let worker = manager.inner.disk_worker.as_ref().unwrap();
    let id = disk_test_id(0);
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let key = CacheIoOperationKey {
        generation: 0,
        id: id.clone(),
        kind: CacheIoOperationKind::Write,
    };
    let submission = worker
        .prepare(
            key.clone(),
            DiskTask::PauseWrite {
                started: started_tx,
                release: release_rx,
                commit: Some(DiskWriteCommit {
                    state: Arc::downgrade(&manager.inner.state),
                    key: key.clone(),
                    reservation_id: 7,
                    armed: true,
                }),
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
                global_layer: id.global_layer,
                logical_bytes: 16,
                host_capacity,
                ticket: ticket.clone(),
            },
        );
        insert_test_record(
            &mut state,
            CacheBlockRecord {
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
}

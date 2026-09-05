#[test]
fn model_reset_surfaces_propagate_paged_clear_failures() {
    use crate::backend::runtime::cache::state::MlxKeyValueState;

    let manager = manager_with_leased_block();
    let layout = eredu_runtime::StateLayout::new(
        eredu_core::LayerSchedule::new(
            1,
            vec![eredu_core::cache::LayerCachePolicy::key_value(
                eredu_core::AttentionPolicy::Full,
                1,
                1,
            )
            .unwrap()],
        )
        .unwrap(),
    )
    .unwrap();
    let mut state = MlxKeyValueState::paged(layout.clone(), manager.clone(), None).unwrap();
    assert!(state.clear().is_err());
    assert_eq!(manager.lock().unwrap().blocks.len(), 1);
}

#[test]
fn disk_worker_coalesces_duplicate_in_flight_reads() {
    let directory = tempfile::tempdir().unwrap();
    let worker = DiskWorker::new(1).unwrap();
    let id = disk_test_id(0);
    let location = missing_location(directory.path(), "missing.safetensors");
    let first = worker
        .prepare_read(3, &id, &location, CacheRepresentation::KeyValue)
        .unwrap();
    let ticket = first.ticket.clone();
    let second = worker
        .prepare_read(3, &id, &location, CacheRepresentation::KeyValue)
        .unwrap();
    assert!(second.inner.joined);
    let second_ticket = second.ticket.clone();
    first.enqueue().unwrap();
    second.enqueue().unwrap();
    assert!(ticket.wait().is_err());
    assert!(second_ticket.wait().is_err());
    assert!(ticket.shares_completion_with(&second_ticket));
    worker.retire(&ticket);
}

#[test]
fn disk_worker_applies_backpressure_only_outside_submission() {
    let directory = tempfile::tempdir().unwrap();
    let worker = DiskWorker::new(1).unwrap();
    let (first_started_tx, first_started_rx) = mpsc::channel();
    let (first_release_tx, first_release_rx) = mpsc::channel();
    let first = worker
        .prepare(
            CacheIoOperationKey {
                generation: 0,
                id: disk_test_id(0),
                kind: CacheIoOperationKind::Read,
            },
            DiskTask::Pause {
                started: first_started_tx,
                release: first_release_rx,
            },
        )
        .unwrap();
    let first_ticket = first.ticket.clone();
    first.enqueue().unwrap();
    first_started_rx.recv().unwrap();

    let (second_started_tx, second_started_rx) = mpsc::channel();
    let (second_release_tx, second_release_rx) = mpsc::channel();
    let second = worker
        .prepare(
            CacheIoOperationKey {
                generation: 0,
                id: disk_test_id(1),
                kind: CacheIoOperationKind::Read,
            },
            DiskTask::Pause {
                started: second_started_tx,
                release: second_release_rx,
            },
        )
        .unwrap();
    let second_ticket = second.ticket.clone();
    second.enqueue().unwrap();

    let third = worker
        .prepare_read(
            0,
            &disk_test_id(2),
            &missing_location(directory.path(), "third.safetensors"),
            CacheRepresentation::KeyValue,
        )
        .unwrap();
    let third_ticket = third.ticket.clone();
    let (outcome_tx, outcome_rx) = mpsc::channel();
    let enqueue_thread = thread::spawn(move || outcome_tx.send(third.enqueue()).unwrap());
    assert!(outcome_rx.recv_timeout(Duration::from_millis(20)).is_err());

    first_release_tx.send(()).unwrap();
    second_started_rx.recv().unwrap();
    let outcome = outcome_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert!(outcome.backpressure);
    assert_eq!(outcome.peak_occupancy, 1);
    second_release_tx.send(()).unwrap();
    enqueue_thread.join().unwrap();
    assert!(matches!(first_ticket.wait().unwrap(), DiskResult::Test));
    assert!(matches!(second_ticket.wait().unwrap(), DiskResult::Test));
    assert!(third_ticket.wait().is_err());
    worker.retire(&first_ticket);
    worker.retire(&second_ticket);
    worker.retire(&third_ticket);
}

#[test]
fn disk_worker_cancels_queued_generation_work() {
    let directory = tempfile::tempdir().unwrap();
    let worker = DiskWorker::new(1).unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let blocker = worker
        .prepare(
            CacheIoOperationKey {
                generation: 8,
                id: disk_test_id(0),
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

    let cancelled = worker
        .prepare_read(
            8,
            &disk_test_id(1),
            &missing_location(directory.path(), "cancelled.safetensors"),
            CacheRepresentation::KeyValue,
        )
        .unwrap();
    let cancelled_ticket = cancelled.ticket.clone();
    cancelled.enqueue().unwrap();
    assert!(cancelled_ticket.cancel());
    assert!(matches!(
        cancelled_ticket.wait(),
        Err(CacheResidencyError::DiskOperationCancelled { generation: 8 })
    ));
    release_tx.send(()).unwrap();
    assert!(matches!(blocker_ticket.wait().unwrap(), DiskResult::Test));
    worker.retire(&blocker_ticket);
    worker.retire(&cancelled_ticket);
}

#[test]
fn cancellation_wakes_a_backpressured_submitter() {
    let directory = tempfile::tempdir().unwrap();
    let worker = DiskWorker::new(1).unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let blocker = worker
        .prepare(
            CacheIoOperationKey {
                generation: 4,
                id: disk_test_id(0),
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

    let queued = worker
        .prepare_read(
            4,
            &disk_test_id(1),
            &missing_location(directory.path(), "queued.safetensors"),
            CacheRepresentation::KeyValue,
        )
        .unwrap();
    let queued_ticket = queued.ticket.clone();
    queued.enqueue().unwrap();
    let blocked = worker
        .prepare_read(
            4,
            &disk_test_id(2),
            &missing_location(directory.path(), "blocked.safetensors"),
            CacheRepresentation::KeyValue,
        )
        .unwrap();
    let blocked_ticket = blocked.ticket.clone();
    let (outcome_tx, outcome_rx) = mpsc::channel();
    let enqueue_thread = thread::spawn(move || outcome_tx.send(blocked.enqueue()).unwrap());
    assert!(outcome_rx.recv_timeout(Duration::from_millis(20)).is_err());

    assert!(blocked_ticket.cancel());
    let outcome = outcome_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    assert!(outcome.backpressure);
    enqueue_thread.join().unwrap();
    assert!(matches!(
        blocked_ticket.wait(),
        Err(CacheResidencyError::DiskOperationCancelled { generation: 4 })
    ));
    release_tx.send(()).unwrap();
    assert!(matches!(blocker_ticket.wait().unwrap(), DiskResult::Test));
    assert!(queued_ticket.wait().is_err());
    worker.retire(&blocker_ticket);
    worker.retire(&queued_ticket);
    worker.retire(&blocked_ticket);
}

#[test]
fn disk_worker_reports_operation_panics_and_keeps_running() {
    let directory = tempfile::tempdir().unwrap();
    let worker = DiskWorker::new(1).unwrap();
    let panicking = worker
        .prepare(
            CacheIoOperationKey {
                generation: 2,
                id: disk_test_id(0),
                kind: CacheIoOperationKind::Read,
            },
            DiskTask::Panic,
        )
        .unwrap();
    let panicking_ticket = panicking.ticket.clone();
    panicking.enqueue().unwrap();
    assert!(matches!(
        panicking_ticket.wait(),
        Err(CacheResidencyError::Runtime(message))
            if message.contains("operation panicked")
    ));
    worker.retire(&panicking_ticket);

    let following = worker
        .prepare_read(
            2,
            &disk_test_id(1),
            &missing_location(directory.path(), "following.safetensors"),
            CacheRepresentation::KeyValue,
        )
        .unwrap();
    let following_ticket = following.ticket.clone();
    following.enqueue().unwrap();
    assert!(following_ticket.wait().is_err());
    worker.retire(&following_ticket);
}

#[test]
fn background_write_failures_surface_on_the_next_foreground_operation() {
    let manager = CacheResidencyManager::new(
        PagedCacheOptions::new(1, 64, 64, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let worker = manager.inner.disk_worker.as_ref().unwrap();
    let id = disk_test_id(0);
    let key = CacheIoOperationKey {
        generation: 0,
        id: id.clone(),
        kind: CacheIoOperationKind::Write,
    };
    let submission = worker.prepare(key.clone(), DiskTask::Panic).unwrap();
    let ticket = submission.ticket.clone();
    insert_test_record(
        &mut manager.lock().unwrap(),
        CacheBlockRecord {
            physical: test_host_writing(test_host_block(), ticket.clone()),
            bytes: 0,
            shapes: [vec![1], vec![1]],
            dtypes: ["Float32".into(), "Float32".into()],
            imported: false,
        },
        false,
        0,
    );
    DiskWriteCommit {
        state: Arc::downgrade(&manager.inner.state),
        key,
        reservation_id: 0,
        armed: true,
    }
    .reconcile(&Err(CacheResidencyError::Runtime(
        "injected asynchronous write failure".into(),
    )));

    let error = manager.set_tail_state(0, 0, 0).unwrap_err();
    assert!(error
        .to_string()
        .contains("injected asynchronous write failure"));
    let report = manager.report().unwrap();
    assert_eq!(report.failures, 1);
    assert_eq!(report.per_layer.len(), 1);
    assert_eq!(report.per_layer[0].global_layer, 0);
    assert_eq!(report.per_layer[0].stats.failures, report.failures);
    worker.retire(&ticket);
}

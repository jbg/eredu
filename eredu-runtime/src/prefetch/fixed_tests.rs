use super::*;
use std::sync::atomic::AtomicUsize;
use std::time::Duration;
fn id(value: &str) -> OffloadUnitId {
    OffloadUnitId::new(value).unwrap()
}

struct ReleaseGate(Arc<(Mutex<bool>, Condvar)>);
impl Drop for ReleaseGate {
    fn drop(&mut self) {
        *self.0.0.lock().unwrap() = true;
        self.0.1.notify_all();
    }
}

#[test]
fn selected_worker_borrows_exact_ids_and_rejects_foreign_units_before_work() {
    let unit = id("unit");
    let pointer = unit.as_str().as_ptr() as usize;
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let worker =
        BackgroundPrefetchWorker::for_units(1, vec![unit], "selected-prefetch", move |value| {
            assert_eq!(value.as_str().as_ptr() as usize, pointer);
            observed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
    let before = worker.report().unwrap();
    assert!(matches!(
        worker.submit(&id("foreign"), false),
        Err(BackgroundPrefetchWorkerError::OutsideDomain)
    ));
    assert!(matches!(
        worker.wait(&id("foreign")),
        Err(BackgroundPrefetchWorkerError::OutsideDomain)
    ));
    assert_eq!(worker.report().unwrap(), before);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    for _ in 0..3 {
        worker.submit(&id("unit"), false).unwrap();
        assert_eq!(
            worker.wait(&id("unit")).unwrap(),
            PrefetchDemandResolution::Ready
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    worker.submit(&id("unit"), true).unwrap();
    assert_eq!(
        worker.wait(&id("unit")).unwrap(),
        PrefetchDemandResolution::Ready
    );
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[test]
fn fixed_notification_stays_coalesced_through_repeated_cancel() {
    // Keep the actual consumer unscheduled while the public submit/cancel
    // methods repeatedly update its one shared wake predicate.
    let storage = PreparedPrefetchStorage::ordinary(1, Some(vec![id("unit")])).unwrap();
    let worker: BackgroundPrefetchWorker = BackgroundPrefetchWorker {
        domain: storage.domain,
        shared: storage.shared,
        worker: None,
        nonblocking_drop: false,
        startup: None,
    };
    for _ in 0..1000 {
        worker.submit(&id("unit"), false).unwrap();
        worker.cancel().unwrap();
    }
    assert!(worker.shared.0.lock().unwrap().wake.pending());
    worker.shared.0.lock().unwrap().wake.consume();
    assert!(!worker.shared.0.lock().unwrap().wake.pending());
    worker.submit(&id("unit"), false).unwrap();
    assert!(worker.shared.0.lock().unwrap().wake.pending());
    worker.shared.0.lock().unwrap().wake.consume();
    worker.cancel().unwrap();
    assert_eq!(worker.report().unwrap().cancelled(), 1001);
}

#[test]
fn selected_terminal_domain_larger_than_queue_keeps_sorted_first_error() {
    let worker = BackgroundPrefetchWorker::for_units(
        1,
        vec![id("c"), id("a"), id("b")],
        "selected-errors",
        |id| Err(id.as_str().to_owned()),
    )
    .unwrap();
    for name in ["c", "b", "a"] {
        worker.submit(&id(name), false).unwrap();
        worker.wait_idle().unwrap();
    }
    let error = worker.cancel().unwrap_err();
    assert!(
        matches!(error,BackgroundPrefetchWorkerError::OperationFailed {id: failed,message} if failed.as_str()=="a" && message=="a")
    );
    assert_eq!(worker.report().unwrap().failed(), 3);
    worker.cancel().unwrap();
    assert!(matches!(
        BackgroundPrefetchWorker::for_units(1, vec![id("a"), id("a")], "duplicate", |_| Ok(())),
        Err(BackgroundPrefetchWorkerError::DuplicateUnit)
    ));
}

#[test]
fn selected_nonjoining_drop_retains_operation_until_active_work_retires() {
    struct Owner(mpsc::Sender<()>);
    impl Drop for Owner {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }
    let (retired_tx, retired_rx) = mpsc::channel();
    let owner = Owner(retired_tx);
    let (entered_tx, entered_rx) = mpsc::channel();
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let active_gate = gate.clone();
    let worker = BackgroundPrefetchWorker::for_units(
        1,
        vec![id("active"), id("future")],
        "selected-drop",
        move |unit| {
            let _retained = &owner;
            entered_tx.send(unit.as_str().to_owned()).unwrap();
            let mut released = active_gate.0.lock().unwrap();
            while !*released {
                released = active_gate.1.wait(released).unwrap();
            }
            Ok(())
        },
    )
    .unwrap()
    .with_nonblocking_drop();
    let _release_on_failure = ReleaseGate(gate.clone());
    worker.submit(&id("active"), false).unwrap();
    assert_eq!(
        entered_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        "active"
    );
    worker.submit(&id("future"), false).unwrap();
    let shared = worker.shared.clone();
    drop(worker);
    assert!(retired_rx.try_recv().is_err());
    *gate.0.lock().unwrap() = true;
    gate.1.notify_all();
    retired_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    let report = shared.0.lock().unwrap().report();
    assert_eq!(report.started(), 1);
    assert_eq!(report.cancelled(), 1);
    assert!(entered_rx.try_recv().is_err());
}

#[test]
fn selected_cancel_fences_active_error_and_never_executes_future_work() {
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let operation_gate = gate.clone();
    let (started_tx, started_rx) = mpsc::channel();
    let worker = Arc::new(
        BackgroundPrefetchWorker::for_units(
            1,
            vec![id("active"), id("future")],
            "selected-cancel",
            move |unit| {
                started_tx.send(unit.as_str().to_owned()).unwrap();
                let mut released = operation_gate.0.lock().unwrap();
                while !*released {
                    released = operation_gate.1.wait(released).unwrap();
                }
                Err("late cancelled result".to_owned())
            },
        )
        .unwrap(),
    );
    let _release_on_failure = ReleaseGate(gate.clone());
    worker.submit(&id("active"), false).unwrap();
    assert_eq!(
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        "active"
    );
    worker.submit(&id("future"), false).unwrap();
    let cancelling = worker.clone();
    let (finished_tx, finished_rx) = mpsc::channel();
    let thread = thread::spawn(move || finished_tx.send(cancelling.cancel()).unwrap());
    let mut state = worker.shared.0.lock().unwrap();
    while state.generation() == 0 {
        let (next, timeout) = worker
            .shared
            .1
            .wait_timeout(state, Duration::from_secs(1))
            .unwrap();
        state = next;
        assert!(
            !timeout.timed_out(),
            "cancellation must publish its generation"
        );
    }
    drop(state);
    assert!(finished_rx.try_recv().is_err());
    *gate.0.lock().unwrap() = true;
    gate.1.notify_all();
    finished_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    thread.join().unwrap();
    assert!(started_rx.try_recv().is_err());
    assert_eq!(
        worker.wait(&id("active")).unwrap(),
        PrefetchDemandResolution::Unscheduled
    );
    let report = worker.report().unwrap();
    assert_eq!(report.started(), 1);
    assert_eq!(report.completed(), 0);
    assert_eq!(report.failed(), 0);
    assert_eq!(report.cancelled(), 2);
}

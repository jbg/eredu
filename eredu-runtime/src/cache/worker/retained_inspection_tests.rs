use std::sync::mpsc;
use super::*;
use crate::cache::CacheIoOperationKind;
use eredu_core::cache::{CacheBlockId, CacheRepresentation};
use std::{sync::atomic::AtomicUsize, time::Duration};

struct Retained(Arc<AtomicUsize>);
impl Drop for Retained {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
struct Task {
    started: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
    _owner: Retained,
}
fn execute(task: Task) -> Result<u64, String> {
    task.started.send(()).unwrap();
    task.release.recv_timeout(Duration::from_secs(10)).unwrap();
    Ok(37)
}
fn discard(_: u64) {}
fn key() -> CacheIoOperationKey {
    CacheIoOperationKey {
        generation: 7,
        id: CacheBlockId {
            session_id: 1,
            global_layer: 0,
            representation: CacheRepresentation::KeyValue,
            start: 0,
            end: 1,
            rank: None,
        },
        kind: CacheIoOperationKind::Read,
    }
}

#[test]
fn nonblocking_retained_work_distinguishes_idle_prepared_and_actual_worker_payload() {
    let worker = CacheIoWorker::new(1, "cache-inspection-payload", execute, discard).unwrap();
    let retired = Arc::new(AtomicUsize::new(0));
    assert_eq!(worker.try_has_retained_work().unwrap(), Some(false));
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let submission = worker
        .prepare(
            key(),
            Task {
                started: started_tx,
                release: release_rx,
                _owner: Retained(Arc::clone(&retired)),
            },
        )
        .unwrap();
    assert_eq!(worker.try_has_retained_work().unwrap(), Some(true));
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert!(matches!(
        started_rx.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    let ticket = submission.ticket.clone();
    submission.enqueue().unwrap();
    started_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    assert_eq!(worker.try_has_retained_work().unwrap(), Some(true));
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    release_tx.send(()).unwrap();
    assert_eq!(ticket.wait().unwrap(), 37);
    ticket.wait_for_task_resources().unwrap();
    worker.retire(&ticket);
    // Task-resource release precedes the outer worker guard's destruction.
    // Wait explicitly for that existing flag; inspection itself never settles it.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while worker.shared.active_payload.load(Ordering::Acquire) {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(retired.load(Ordering::SeqCst), 1);
    assert_eq!(worker.try_has_retained_work().unwrap(), Some(false));
}

#[test]
fn nonblocking_retained_work_reports_busy_before_registry_owner_releases_lock() {
    let worker =
        Arc::new(CacheIoWorker::new(1, "cache-inspection-busy", execute, discard).unwrap());
    let (locked_tx, locked_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let other = Arc::clone(&worker);
    let thread = std::thread::spawn(move || {
        let _guard = other.shared.in_flight.lock().unwrap();
        locked_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    });
    locked_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    let observed = worker.try_has_retained_work();
    release_tx.send(()).unwrap();
    thread.join().unwrap();
    assert_eq!(observed.unwrap(), None);
    assert_eq!(worker.try_has_retained_work().unwrap(), Some(false));
}

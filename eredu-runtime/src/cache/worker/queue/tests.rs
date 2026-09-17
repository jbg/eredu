use super::super::test_support::context;
use super::*;
use crate::cache::CacheIoOperationKind;
use eredu_core::cache::{CacheBlockId, CacheRepresentation};
use std::{sync::mpsc, time::Duration};

enum Task {
    Pause(mpsc::Sender<()>, mpsc::Receiver<()>, Arc<()>),
    Value(Arc<()>),
}
fn execute(task: Task) -> Result<u64, String> {
    match task {
        Task::Pause(started, release, retained) => {
            started.send(()).unwrap();
            release.recv().unwrap();
            drop(retained);
            Ok(1)
        }
        Task::Value(_) => panic!("cancelled queued task must not execute"),
    }
}
fn discard(_: u64) {}
fn key(start: i64) -> CacheIoOperationKey {
    CacheIoOperationKey {
        generation: 5,
        id: CacheBlockId {
            session_id: 6,
            global_layer: 1,
            representation: CacheRepresentation::KeyValue,
            start,
            end: start + 1,
            rank: None,
        },
        kind: CacheIoOperationKind::Read,
    }
}

#[test]
fn paid_fifo_retains_detached_cancelled_payloads_and_source_until_actual_receiver_retirement() {
    let (context, account) = context();
    let worker = CacheIoWorker::new(1, "paid-cache-fifo", execute, discard)
        .unwrap()
        .with_nonblocking_drop();
    let foreign = CacheIoWorker::new(1, "foreign-cache-fifo", execute, discard).unwrap();
    let queue = worker.prepare_queue(&context).unwrap();
    let failure = queue.install(&foreign).unwrap_err();
    assert_eq!(failure.cause(), CacheIoRegistryRefusal::ForeignWorker);
    let (_, queue) = failure.into_parts();
    drop(queue.install(&worker).unwrap());
    drop(context);
    let first = Arc::new(());
    let first_weak = Arc::downgrade(&first);
    let second = Arc::new(());
    let second_weak = Arc::downgrade(&second);
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let active = worker
        .prepare(key(0), Task::Pause(started_tx, release_rx, first))
        .unwrap();
    let active_ticket = active.ticket.clone();
    active.enqueue().unwrap();
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    let queued = worker.prepare(key(1), Task::Value(second)).unwrap();
    let queued_ticket = queued.ticket.clone();
    queued.enqueue().unwrap();
    assert!(queued_ticket.cancel());
    // Observe the same thread's real termination after testing nonblocking
    // owner drop, without treating cancellation or publication as retirement.
    let handle = worker.handle.lock().unwrap().take().unwrap();
    drop((foreign, worker));
    let retained = first_weak.upgrade().is_some() && second_weak.upgrade().is_some();
    let funded = !account.retired.load(Ordering::SeqCst);
    release_tx.send(()).unwrap();
    active_ticket.wait_for_task_resources().unwrap();
    queued_ticket.wait_for_task_resources().unwrap();
    handle.join().unwrap();
    assert!(retained);
    assert!(funded);
    assert!(first_weak.upgrade().is_none());
    assert!(second_weak.upgrade().is_none());
    assert!(matches!(
        queued_ticket.wait(),
        Err(CacheIoWorkerError::Cancelled { generation: 5 })
    ));
    assert!(
        account.retired.load(Ordering::SeqCst),
        "queue backing outlived its last receiver/sender owner"
    );
}

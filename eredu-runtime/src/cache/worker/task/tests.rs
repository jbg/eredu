use super::super::test_support::context;
use super::*;
use crate::cache::{CacheIoOperationKind, CacheTableCapacityError};
use eredu_core::cache::{CacheBlockId, CacheRepresentation};

fn key(start: i64) -> CacheIoOperationKey {
    CacheIoOperationKey {
        generation: 17,
        id: CacheBlockId {
            session_id: 4,
            global_layer: 2,
            representation: CacheRepresentation::KeyValue,
            start,
            end: start + 1,
            rank: None,
        },
        kind: CacheIoOperationKind::Read,
    }
}
struct Payload {
    value: u64,
    marker: Arc<()>,
}
impl Clone for Payload {
    fn clone(&self) -> Self {
        panic!("paid borrowed result must never clone payload")
    }
}
fn execute(value: Payload) -> Result<Payload, String> {
    Ok(value)
}
fn discard(_: Payload) {}
fn payload(value: u64) -> (Payload, std::sync::Weak<()>) {
    let marker = Arc::new(());
    let weak = Arc::downgrade(&marker);
    (Payload { value, marker }, weak)
}

#[test]
fn paid_task_preserves_foreign_full_join_and_borrowed_output_until_last_ticket_retirement() {
    let (registry_context, registry_account) = context();
    let (task_context, task_account) = context();
    let worker = CacheIoWorker::new(1, "paid-disk-task", execute, discard).unwrap();
    let foreign = CacheIoWorker::new(1, "foreign-disk-task", execute, discard).unwrap();
    drop(
        worker
            .prepare_registry(1, &registry_context)
            .unwrap()
            .install(&worker)
            .unwrap(),
    );
    drop(
        worker
            .prepare_queue(&registry_context)
            .unwrap()
            .install(&worker)
            .unwrap(),
    );
    let slot = worker.prepare_task_slot(key(0), &task_context).unwrap();
    let charged = *task_account.remaining.lock().unwrap();
    let (value, retained) = payload(41);
    let prepared = slot.bind(value);
    assert_eq!(
        *task_account.remaining.lock().unwrap(),
        charged,
        "binding uses the paid shell"
    );
    let error = prepared
        .prepare(&foreign)
        .err()
        .expect("foreign worker refused");
    assert!(matches!(error.cause(), CacheIoTaskRefusal::ForeignWorker));
    assert!(retained.upgrade().is_some());
    let (_, prepared) = error.into_parts();
    let first = prepared.prepare(&worker).unwrap();
    let first_ticket = first.ticket.clone();

    let (value, unused) = payload(99);
    let mut joined = worker
        .prepare_task(key(0), value, &task_context)
        .unwrap()
        .prepare(&worker)
        .unwrap();
    assert!(joined.joined);
    assert!(joined.ticket.shares_completion_with(&first_ticket));
    joined.joined_task_mut().unwrap().value = 100;
    let joined_ticket = joined.ticket.clone();
    joined.enqueue().unwrap();
    assert!(unused.upgrade().is_none());

    let (value, refused) = payload(72);
    let error = worker
        .prepare_task(key(1), value, &task_context)
        .unwrap()
        .prepare(&worker)
        .err()
        .expect("finite registry refused");
    assert!(matches!(
        error.cause(),
        CacheIoTaskRefusal::Worker(CacheIoWorkerError::Execution(
            CacheIoExecutionStateError::RegistryCapacity(CacheTableCapacityError::Exhausted)
        ))
    ));
    assert!(refused.upgrade().is_some());
    drop(error);
    assert!(refused.upgrade().is_none());
    let next_slot = worker.prepare_task_slot(key(0), &task_context).unwrap();
    let spent = *task_account.remaining.lock().unwrap();
    drop(task_context);
    first.enqueue().unwrap();
    first_ticket.with_result(|result| {
        let result = result.unwrap();
        assert_eq!(result.value, 41);
        assert!(Arc::ptr_eq(&result.marker, &retained.upgrade().unwrap()));
    });
    first_ticket.wait_for_task_resources().unwrap();
    let (value, second_retained) = payload(42);
    let next = next_slot.bind(value).prepare(&worker).unwrap();
    assert!(
        !next.joined,
        "retired occurrence cannot replace a fresh paid read"
    );
    assert!(!next.ticket.shares_completion_with(&first_ticket));
    let next_ticket = next.ticket.clone();
    next.enqueue().unwrap();
    next_ticket.with_result(|value| assert_eq!(value.unwrap().value, 42));
    next_ticket.wait_for_task_resources().unwrap();
    drop(next_ticket);
    drop((worker, foreign, registry_context));
    assert!(second_retained.upgrade().is_none());
    assert!(!registry_account.retired.load(Ordering::SeqCst));
    assert!(!task_account.retired.load(Ordering::SeqCst));
    assert_eq!(*task_account.remaining.lock().unwrap(), spent);
    drop(first_ticket);
    assert!(retained.upgrade().is_some());
    assert!(!task_account.retired.load(Ordering::SeqCst));
    drop(joined_ticket);
    assert!(retained.upgrade().is_none());
    assert!(task_account.retired.load(Ordering::SeqCst));
    assert!(registry_account.retired.load(Ordering::SeqCst));
}

#[test]
fn paid_task_cannot_join_an_ordinary_completion_in_prepared_registry() {
    let (context, _) = context();
    let worker = CacheIoWorker::new(1, "paid-task-ordinary-join", execute, discard).unwrap();
    drop(
        worker
            .prepare_registry(1, &context)
            .unwrap()
            .install(&worker)
            .unwrap(),
    );
    drop(
        worker
            .prepare_queue(&context)
            .unwrap()
            .install(&worker)
            .unwrap(),
    );
    let ordinary = worker.prepare(key(0), payload(1).0).unwrap();
    let paid = worker.prepare_task(key(0), payload(2).0, &context).unwrap();
    let error = paid
        .prepare(&worker)
        .err()
        .expect("ordinary task grants no prepared completion");
    assert!(matches!(
        error.cause(),
        CacheIoTaskRefusal::Worker(CacheIoWorkerError::Execution(
            CacheIoExecutionStateError::RegistryCapacity(CacheTableCapacityError::Unprepared)
        ))
    ));
    drop((error, ordinary));
}

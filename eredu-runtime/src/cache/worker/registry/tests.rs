use super::super::test_support::context;
use super::*;
use crate::cache::{CacheIoOperationKind, CacheTableCapacityError};
use eredu_core::cache::{CacheBlockId, CacheRepresentation};

fn key(start: i64) -> CacheIoOperationKey {
    CacheIoOperationKey {
        generation: 11,
        id: CacheBlockId {
            session_id: 3,
            global_layer: 2,
            representation: CacheRepresentation::KeyValue,
            start,
            end: start + 1,
            rank: None,
        },
        kind: CacheIoOperationKind::Read,
    }
}
fn execute(value: u64) -> Result<u64, String> {
    Ok(value)
}
fn discard(_: u64) {}

#[test]
fn finite_registry_preserves_atomic_join_refusal_reuse_and_escaped_ticket_custody() {
    let (context, account) = context();
    let worker = CacheIoWorker::new(1, "finite-cache-registry", execute, discard).unwrap();
    let foreign = CacheIoWorker::new(1, "foreign-cache-registry", execute, discard).unwrap();
    let pending = worker.prepare(key(0), 7).unwrap();
    let destination = worker.prepare_registry(1, &context).unwrap();
    let failure = destination.install(&foreign).unwrap_err();
    assert_eq!(failure.cause(), CacheIoRegistryRefusal::ForeignWorker);
    let (_, destination) = failure.into_parts();
    let failure = destination.install(&worker).unwrap_err();
    assert_eq!(failure.cause(), CacheIoRegistryRefusal::Busy);
    let (_, destination) = failure.into_parts();
    drop(pending);
    drop(destination.install(&worker).unwrap());
    // Only the registries have been paid here; task/result/channel constructors
    // remain ordinary until their separate source-qualified producer joins.
    let first = worker.prepare(key(1), 9).unwrap();
    let escaped = first.ticket.clone();
    let joined = worker.prepare(key(1), 10).unwrap();
    assert!(joined.joined);
    assert!(joined.ticket.shares_completion_with(&escaped));
    match worker.prepare(key(2), 11) {
        Err(CacheIoWorkerError::Execution(CacheIoExecutionStateError::RegistryCapacity(
            CacheTableCapacityError::Exhausted,
        ))) => {}
        _ => panic!("one retained-operation slot must reject a different key"),
    }
    drop(joined);
    first.enqueue().unwrap();
    assert_eq!(escaped.wait().unwrap(), 9);
    escaped.wait_for_task_resources().unwrap();
    worker.retire(&escaped);
    let reused = worker.prepare(key(2), 12).unwrap();
    let reused_ticket = reused.ticket.clone();
    reused.enqueue().unwrap();
    assert_eq!(reused_ticket.wait().unwrap(), 12);
    reused_ticket.wait_for_task_resources().unwrap();
    drop(reused_ticket);
    drop((foreign, worker, context));
    assert!(
        !account.retired.load(Ordering::SeqCst),
        "escaped exact ticket keeps registry custody"
    );
    drop(escaped);
    assert!(account.retired.load(Ordering::SeqCst));
}

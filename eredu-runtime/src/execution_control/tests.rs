use super::*;

#[test]
fn cancellation_after_running_admission_does_not_invent_a_prediction() {
    let mut lifecycle = GenerationLifecycle::default();
    assert!(lifecycle.cancel_without_prediction().is_err());
    lifecycle.begin_prediction().unwrap();
    lifecycle.cancel_without_prediction().unwrap();
    assert_eq!(lifecycle.next_prediction(), 0);
    assert_eq!(lifecycle.status(), GenerationStatus::Cancelled);
    assert!(lifecycle.checkpoint().is_err());
    assert!(lifecycle.begin_prediction().is_err());
}

#[test]
fn lifecycle_only_snapshots_completed_and_delivered_boundaries() {
    let mut run = GenerationLifecycle::default();
    let initial = run.checkpoint().unwrap();
    run.begin_prediction().unwrap();
    assert!(run.pause().is_err());
    assert!(run.cancel().is_err());
    assert!(run.checkpoint().is_err());
    assert!(run.restore(&initial).is_err());
    assert!(run.begin_prediction().is_err());
    run.complete_prediction(None).unwrap();
    assert_eq!(run.status(), GenerationStatus::Paused);
    assert_eq!(run.next_prediction(), 1);
    let saved = run.checkpoint().unwrap();
    let mut child = GenerationLifecycle::fork(&saved);
    child.begin_prediction().unwrap();
    child.complete_prediction(Some(FinishReason::Eos)).unwrap();
    let terminal = child.checkpoint().unwrap();
    assert!(child.begin_prediction().is_err());
    assert_eq!(run.next_prediction(), 1);
    run.restore(&terminal).unwrap();
    assert_eq!(run.status(), GenerationStatus::Completed);
    assert_eq!(run.finish_reason(), Some(FinishReason::Eos));
    assert!(run.begin_prediction().is_err());
    run.restore(&saved).unwrap();
    assert_eq!(run.status(), GenerationStatus::Paused);
    assert_eq!(run.next_prediction(), 1);
    assert_eq!(run.epoch(), 2);
    assert_eq!(run.finish_reason(), None);
    run.restore(&initial).unwrap();
    assert_eq!(run.next_prediction(), 0);
    run.cancel().unwrap();
    assert!(run.checkpoint().is_err());
    assert!(run.restore(&saved).is_err());
    assert_eq!(run.finish_reason(), Some(FinishReason::Cancelled));
    child.fail();
    assert!(child.checkpoint().is_err());
    assert!(child.restore(&saved).is_err());
}

#[test]
fn pause_requests_cross_threads_without_moving_native_owners_or_cancelling() {
    let local_owner = Rc::new(());
    let handle = GenerationControlHandle::default();
    let worker_handle = handle.clone();
    std::thread::spawn(move || worker_handle.request_pause())
        .join()
        .unwrap();
    assert!(handle.pause_requested());
    assert!(!handle.cancellation().is_cancelled());
    handle.acknowledge_resume();
    assert!(!handle.pause_requested());
    handle.cancel();
    handle.acknowledge_resume();
    assert!(handle.cancellation().is_cancelled());
    assert_eq!(Rc::strong_count(&local_owner), 1);
}

fn budget() -> SnapshotBudget {
    SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 1,
        retained_bytes: 100,
        cumulative_copy_bytes: 200,
    })
}

#[test]
fn reservations_bound_live_handles_and_never_refund_consumed_copy_work() {
    let budget = budget();
    let cost = Some(SnapshotEstimate {
        retained_bytes: 40,
        copy_bytes: 40,
    });
    let saved = budget
        .reserve(SnapshotResourceKind::Snapshot, cost)
        .unwrap();
    let another_handle = saved.clone();
    assert!(budget
        .reserve(SnapshotResourceKind::Snapshot, cost)
        .is_err());
    let child = budget.reserve(SnapshotResourceKind::Branch, cost).unwrap();
    assert_eq!(budget.usage().retained_bytes, 80);
    assert!(budget.reserve(SnapshotResourceKind::Restore, cost).is_err());
    drop(saved);
    assert_eq!(budget.usage().snapshots, 1);
    drop(another_handle);
    assert_eq!(budget.usage().snapshots, 0);
    // A failed native copy drops its provisional lease; cumulative work stays charged.
    let failed = budget.reserve(SnapshotResourceKind::Restore, cost).unwrap();
    drop(failed);
    assert_eq!(budget.usage().retained_bytes, 40);
    assert_eq!(budget.usage().cumulative_copy_bytes, 120);
    drop(child);
    let next = budget
        .reserve(SnapshotResourceKind::Snapshot, cost)
        .unwrap();
    let final_copy = budget.reserve(SnapshotResourceKind::Restore, cost).unwrap();
    drop(final_copy);
    assert!(budget.reserve(SnapshotResourceKind::Restore, cost).is_err());
    drop(next);
    assert_eq!(
        budget.usage(),
        SnapshotUsage {
            cumulative_copy_bytes: 200,
            ..Default::default()
        }
    );
}

#[test]
fn unavailable_or_overflowing_estimates_fail_before_reservation() {
    let budget = budget();
    assert!(matches!(
        budget.reserve(SnapshotResourceKind::Snapshot, None),
        Err(ExecutionControlError::UnknownEstimate)
    ));
    assert_eq!(budget.usage(), SnapshotUsage::default());
    let saved = budget
        .reserve(
            SnapshotResourceKind::Snapshot,
            Some(SnapshotEstimate {
                retained_bytes: 1,
                copy_bytes: 1,
            }),
        )
        .unwrap();
    let before = budget.usage();
    assert!(matches!(
        budget.reserve(
            SnapshotResourceKind::Restore,
            Some(SnapshotEstimate {
                retained_bytes: u64::MAX,
                copy_bytes: 0,
            })
        ),
        Err(ExecutionControlError::Overflow)
    ));
    assert_eq!(budget.usage(), before);
    drop(saved);
}

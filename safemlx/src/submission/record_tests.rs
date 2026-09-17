use super::*;
use crate::{transforms::async_eval_with_event, Array, Device, DeviceType, Stream};
use std::{cell::Cell, sync::mpsc, time::Duration};
thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.with(|n| n.set(n.get() + 1));
}
struct Hook;
impl Hook {
    fn install() -> Self {
        runtime_lock::register_housekeeping_hook(housekeeping);
        HOUSEKEEPING.with(|n| n.set(0));
        Self
    }
}
impl Drop for Hook {
    fn drop(&mut self) {
        runtime_lock::unregister_housekeeping_hook(housekeeping);
    }
}
fn settle(scope: &SubmissionScope) -> SubmissionStatus {
    let end = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let status = scope.progress();
        if status.is_settled() {
            return status;
        }
        assert!(
            std::time::Instant::now() < end,
            "native scope did not settle: {status:?}"
        );
        std::thread::yield_now();
    }
}
#[test]
fn live_empty_child_is_record_quiescent_without_lifetime_settlement_or_housekeeping() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let source = Array::from_slice(&[2_f32, -3., 5., 7.], &[4]);
    let lazy = source.square(&stream).unwrap();
    let mut outer = SubmissionScope::begin().unwrap();
    let mut child = SubmissionScope::begin().unwrap();
    outer.seal();
    let _hook = Hook::install();
    for scope in [&outer, &child] {
        let records = scope.accepted_records();
        assert_eq!(records.activity(), SubmissionRecordActivity::Quiescent);
        assert!(!records.failed() && !records.blocked());
    }
    assert!(!outer.status().is_settled());
    assert_eq!(child.status().activity(), SubmissionActivity::None);
    assert!(SubmissionScope::native_control_bytes() > 0);
    assert!(lazy.try_metadata_snapshot().unwrap().allocation().is_none());
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    child.seal();
    assert!(outer.status().is_settled());
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
}
#[test]
fn record_observation_does_not_wait_for_foreign_runtime_or_run_housekeeping() {
    let scope = SubmissionScope::begin().unwrap();
    let _hook = Hook::install();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        entered_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
    });
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let records = scope.accepted_records();
    let bytes = SubmissionScope::native_control_bytes();
    let sent = release_tx.send(()).is_ok();
    let held_until_release = worker.join().unwrap();
    assert!(
        sent && held_until_release,
        "cold observation waited for runtime"
    );
    assert!(bytes > 0);
    assert_eq!(records.activity(), SubmissionRecordActivity::Quiescent);
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
}
#[test]
fn real_cpu_completion_keeps_live_child_structural_pending_separate() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let source = Array::from_slice(&[2_f32, -3., 5., 7.], &[4]);
    let mut outer = SubmissionScope::begin().unwrap();
    let mut child = SubmissionScope::begin().unwrap();
    let output = source.square(&stream).unwrap();
    let completion = async_eval_with_event([&output]).unwrap();
    completion.synchronize().unwrap();
    assert!(settle(&child).is_settled());
    outer.seal();
    let _hook = Hook::install();
    for scope in [&outer, &child] {
        let records = scope.accepted_records();
        assert_eq!(records.activity(), SubmissionRecordActivity::Quiescent);
        assert!(!records.failed() && !records.blocked());
    }
    assert!(!outer.status().is_settled());
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    child.seal();
    assert!(outer.status().is_settled());
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    drop(_hook);
    assert_eq!(
        output.evaluated().unwrap().as_slice::<f32>(),
        &[4., 9., 25., 49.]
    );
}
#[test]
fn failed_observation_stays_unknown_and_failure_flags_are_not_settlement() {
    let mut scope = SubmissionScope::begin().unwrap();
    scope.observation_failed = true;
    let _hook = Hook::install();
    let records = scope.accepted_records();
    assert_eq!(records.activity(), SubmissionRecordActivity::Unobservable);
    assert!(records.failed() && records.blocked());
    let failed = SubmissionRecordStatus::from_raw(safemlx_sys::mlx_submission_record_status {
        pending: false,
        failed: true,
        blocked: true,
    });
    assert_eq!(failed.activity(), SubmissionRecordActivity::Quiescent);
    assert!(failed.failed() && failed.blocked());
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
}
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[test]
fn actual_gpu_and_stream_wait_records_reach_existing_terminal_transition() {
    let device = Device::new(DeviceType::Gpu, 0);
    let producer = Stream::new_with_device(&device);
    let consumer = Stream::new_with_device(&device);
    let mut outer = SubmissionScope::begin().unwrap();
    let mut child = SubmissionScope::begin().unwrap();
    let value = Array::ones::<f32>(&[8], &producer).unwrap();
    let completion = async_eval_with_event([&value]).unwrap();
    completion.synchronize().unwrap();
    completion.wait_on(&consumer).unwrap();
    let status = settle(&child);
    assert!(!status.failed() && !status.blocked());
    outer.seal();
    let _hook = Hook::install();
    assert_eq!(
        outer.accepted_records().activity(),
        SubmissionRecordActivity::Quiescent
    );
    assert_eq!(
        child.accepted_records().activity(),
        SubmissionRecordActivity::Quiescent
    );
    assert!(!outer.status().is_settled());
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    child.seal();
    assert!(outer.status().is_settled());
}

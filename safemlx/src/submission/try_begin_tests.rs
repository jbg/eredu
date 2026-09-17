use super::*;
use crate::{transforms::async_eval_with_event, Array, Device, DeviceType, Stream};
use std::{cell::Cell, sync::mpsc, time::Duration};

thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.with(|calls| calls.set(calls.get() + 1));
}
struct Hook;
impl Hook {
    fn install() -> Self {
        runtime_lock::register_housekeeping_hook(housekeeping);
        HOUSEKEEPING.with(|calls| calls.set(0));
        Self
    }
}
impl Drop for Hook {
    fn drop(&mut self) {
        runtime_lock::unregister_housekeeping_hook(housekeeping);
    }
}

#[test]
fn busy_try_begin_does_not_wait_enter_scope_run_housekeeping_or_evaluate() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let source = Array::from_slice(&[2_f32, -3., 5., 7.], &[4]);
    let lazy = source.square(&stream).unwrap();
    let mut outer = SubmissionScope::begin().unwrap();
    let before = outer.status();
    let _hook = Hook::install();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        entered_tx.send(()).unwrap();
        // Broken blocking code reports a failure instead of hanging the suite.
        release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
    });
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let result = SubmissionScope::try_begin();
    let released = release_tx.send(()).is_ok();
    let held_until_release = worker.join().unwrap();
    assert!(
        released && held_until_release,
        "try_begin waited for the lock"
    );
    assert!(matches!(result, Err(SubmissionScopeBeginError::Busy)));
    assert_eq!(outer.status(), before);
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    assert!(lazy.try_metadata_snapshot().unwrap().allocation().is_none());
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    outer.seal();
}

#[test]
fn successful_try_begin_skips_housekeeping_and_tracks_real_cpu_work() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let source = Array::from_slice(&[2_f32, -3., 5., 7.], &[4]);
    let _hook = Hook::install();
    let mut scope = SubmissionScope::try_begin().unwrap();
    assert_eq!(scope.status().activity(), SubmissionActivity::None);
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    let output = source.square(&stream).unwrap();
    let event = async_eval_with_event([&output]).unwrap();
    scope.seal();
    assert!(scope.status().has_work());
    event.synchronize().unwrap();
    let status = scope.progress();
    assert!(status.is_settled() && !status.failed() && !status.blocked());
    assert_eq!(
        output.evaluated().unwrap().as_slice::<f32>(),
        &[4., 9., 25., 49.]
    );
}

#[test]
fn successful_try_begin_retains_real_parent_child_scope_order() {
    let mut outer = SubmissionScope::try_begin().unwrap();
    let mut inner = SubmissionScope::try_begin().unwrap();
    outer.seal();
    assert!(!outer.status().is_settled());
    assert_eq!(inner.status().activity(), SubmissionActivity::None);
    inner.seal();
    assert_eq!(outer.status().activity(), SubmissionActivity::None);
}

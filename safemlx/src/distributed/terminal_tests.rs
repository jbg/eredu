use super::*;
use crate::{Device, DeviceType, SubmissionScope};
use std::{cell::Cell, sync::mpsc, time::Duration};

thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.set(HOUSEKEEPING.get() + 1);
}
struct Hook;
impl Drop for Hook {
    fn drop(&mut self) {
        runtime_lock::unregister_housekeeping_hook(housekeeping);
    }
}
fn terminal(error: &Exception) -> bool {
    std::error::Error::source(error).is_some_and(|source| source.is::<TerminalGroup>())
}

#[test]
fn canonical_terminal_fence_isolated() {
    const FLAG: &str = "EREDU_TERMINAL_GROUP_CHILD";
    if std::env::var_os(FLAG).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "distributed::terminal_tests::canonical_terminal_fence_isolated",
                "--nocapture",
            ])
            .env(FLAG, "1")
            .env_remove("MLX_HOSTFILE")
            .env_remove("MLX_RANK")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    let group = init(false, Backend::Ring).unwrap();
    assert_eq!(group.size(), 1);
    let alias = group.clone();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let input = Array::from_slice(&[1.25_f32, -2.5, 3.75], &[3]);
    let lazy = input.square(&stream).unwrap();
    let prior = all_sum(&lazy, &group, &stream).unwrap();
    let mut scope = SubmissionScope::begin().unwrap();
    let before = scope.status();
    runtime_lock::register_housekeeping_hook(housekeeping);
    let _hook = Hook;
    HOUSEKEEPING.set(0);
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        entered_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
    });
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    group.mark_terminal_submission();
    assert!(alias.terminal_submission());
    assert_eq!(group.check_submission_available(), Err(TerminalGroup));
    // Every entry rejects before trying the held runtime lock or output guard.
    let results = [
        all_sum(&input, &alias, &stream),
        all_max(&input, &alias, &stream),
        all_min(&input, &alias, &stream),
        all_gather(&input, &alias, &stream),
        all_to_all_v(&input, &[3], &[3], &alias, &stream),
        sum_scatter(&input, &alias, &stream),
        send(&input, 0, &alias, &stream),
        recv(&[3], Dtype::Float32, 0, &alias, &stream),
        recv_like(&input, 0, &alias, &stream),
    ];
    let split = alias.split(0, None);
    let transport = alias.communication_stream();
    let released = release_tx.send(()).is_ok();
    let held = worker.join().unwrap();
    assert!(
        released && held,
        "terminal preflight waited for runtime lock"
    );
    for result in results {
        assert!(terminal(&result.unwrap_err()));
    }
    assert!(terminal(&split.unwrap_err()));
    assert!(terminal(&transport.unwrap_err()));
    assert_eq!(HOUSEKEEPING.get(), 0);
    assert_eq!(scope.status(), before);
    assert!(lazy.try_metadata_snapshot().unwrap().allocation().is_none());
    assert_eq!(HOUSEKEEPING.get(), 0);
    drop(_hook);
    let repeated = init(false, Backend::Ring).unwrap();
    assert!(
        !repeated.shares_native_handle(&group),
        "fresh C wrapper expected"
    );
    assert!(
        repeated.terminal_submission(),
        "cached native identity was bypassed"
    );
    assert!(terminal(&all_sum(&input, &repeated, &stream).unwrap_err()));
    scope.seal();
    // A preexisting noncommunicating singleton graph remains evaluable; marking
    // does not retroactively cancel/complete/free accepted graph ownership.
    assert_eq!(
        prior.evaluated().unwrap().as_slice::<f32>(),
        &[1.5625, 6.25, 14.0625]
    );
    let empty = Array::from_slice::<f32>(&[], &[0]);
    assert!(terminal(
        &all_to_all_v(&empty, &[0], &[0], &repeated, &stream).unwrap_err()
    ));
}

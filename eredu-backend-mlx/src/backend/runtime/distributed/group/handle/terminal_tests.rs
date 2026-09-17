use super::*;
use crate::backend::nn::shared::MlxNeuralBackend;
use eredu_runtime::{FailureAgreementBackend, TerminalCommunicationBackend};
use std::{sync::mpsc, time::Duration};

fn terminal(mut error: &(dyn std::error::Error + 'static)) -> bool {
    loop {
        if error.is::<safemlx::distributed::TerminalGroup>() {
            return true;
        }
        let Some(source) = error.source() else {
            return false;
        };
        error = source;
    }
}

#[test]
fn terminal_incarnation_rejects_status_before_native_setup_isolated() {
    const FLAG: &str = "EREDU_TERMINAL_STATUS_CHILD";
    if std::env::var_os(FLAG).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "backend::runtime::distributed::group::handle::terminal_tests::terminal_incarnation_rejects_status_before_native_setup_isolated", "--nocapture"])
            .env(FLAG,"1").env_remove("MLX_HOSTFILE").env_remove("MLX_RANK")
            .status().unwrap();
        assert!(status.success());
        return;
    }
    let native = native::init(false, native::Backend::Ring).unwrap();
    let group = Group::uncontracted(&native);
    let logical = group.logical_subgroup(&[0]).unwrap();
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let reinitialized = native::init(false, native::Backend::Ring).unwrap();
    let wrapper = Group::uncontracted(&reinitialized);
    reset_native_collective_submissions();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = safemlx::RuntimeCallDeadline::new(Duration::from_secs(5))
            .unwrap()
            .enter()
            .unwrap();
        entered_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
    });
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    MlxNeuralBackend::mark_terminal_submission(&logical);
    assert!(reinitialized.terminal_submission());
    let first = MlxNeuralBackend::agree_success(true, &group, &stream);
    let second = MlxNeuralBackend::agree_success(false, &wrapper, &stream);
    let third = MlxNeuralBackend::agree_success(true, &logical, &stream);
    let released = release_tx.send(()).is_ok();
    let held = worker.join().unwrap();
    assert!(
        released && held,
        "status setup acquired runtime before terminal check"
    );
    for result in [first, second, third] {
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("terminal status accepted"),
        };
        assert!(terminal(&error));
    }
    assert_eq!(native_collective_submissions(), 0);
    assert_eq!(contracted_collective_submissions(), 0);
}

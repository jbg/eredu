use super::*;
use crate::{Array, Device, DeviceType, Stream};
use std::{
    cell::Cell,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
    time::Duration,
};

thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.with(|count| count.set(count.get() + 1));
}
struct Hook;
impl Hook {
    fn install() -> Self {
        runtime_lock::register_housekeeping_hook(housekeeping);
        HOUSEKEEPING.with(|count| count.set(0));
        Self
    }
}
impl Drop for Hook {
    fn drop(&mut self) {
        runtime_lock::unregister_housekeeping_hook(housekeeping);
    }
}

#[test]
fn retirement_pass_returns_busy_for_foreign_runtime_lock_without_housekeeping() {
    let _hook = Hook::install();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::try_enter_for_recovery().expect("idle runtime");
        entered_tx.send(()).unwrap();
        // A broken blocking implementation eventually returns so the assertion
        // reports failure rather than leaving the test process deadlocked.
        release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
    });
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    let result = try_retire_completed_submissions();
    let released = release_tx.send(()).is_ok();
    let waited_for_release = worker.join().unwrap();
    assert!(
        released && waited_for_release,
        "retirement waited for the held runtime lock"
    );
    assert_eq!(result.unwrap(), SubmissionRetirement::Busy);
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    assert_eq!(
        try_retire_completed_submissions().unwrap(),
        SubmissionRetirement::CompleteSnapshot
    );
}

#[test]
fn retirement_pass_leaves_lazy_array_unknown_and_scope_status_unchanged() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let source = Array::from_slice(&[2_f32, -3., 5., 7.], &[2, 2]);
    let mut scope = SubmissionScope::begin().unwrap();
    let lazy = source.square(&stream).unwrap();
    scope.seal();
    let before = scope.status();
    let source_info = source.try_metadata_snapshot().unwrap().allocation();
    assert!(lazy.try_metadata_snapshot().unwrap().allocation().is_none());
    let _hook = Hook::install();
    assert_eq!(
        try_retire_completed_submissions().unwrap(),
        SubmissionRetirement::CompleteSnapshot
    );
    assert_eq!(scope.status(), before);
    assert_eq!(
        source.try_metadata_snapshot().unwrap().allocation(),
        source_info
    );
    assert!(lazy.try_metadata_snapshot().unwrap().allocation().is_none());
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
}

struct Owner(Arc<AtomicUsize>);
impl Drop for Owner {
    fn drop(&mut self) {
        assert!(can_reclaim_submission_resources());
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn retirement_pass_preserves_actual_alias_charge_and_leaves_host_reclaim_explicit() {
    // Cache eviction synchronizes every native CPU stream. Other tests
    // deliberately terminate a worker; this fixture needs its own live
    // process to verify the explicit eviction/reclaim boundary independently.
    const CHILD: &str = "SAFEMLX_EXPLICIT_RETIREMENT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let module = module_path!().split_once("::").unwrap().1;
        let name = format!(
            "{module}::retirement_pass_preserves_actual_alias_charge_and_leaves_host_reclaim_explicit"
        );
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &name, "--nocapture", "--test-threads=1"])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains("1 passed; 0 failed;"),
            "isolated explicit retirement test failed:\n{stdout}\n{stderr}"
        );
        return;
    }
    let source = Array::from_slice(&[11_i32, -13, 17, 19], &[4]);
    let alias = source.clone();
    let allocation = source
        .try_metadata_snapshot()
        .unwrap()
        .allocation()
        .unwrap();
    let retired = Arc::new(AtomicUsize::new(0));
    source
        .retain_allocation_owner(Owner(retired.clone()))
        .unwrap();
    drop(source);
    let _hook = Hook::install();
    assert_eq!(
        try_retire_completed_submissions().unwrap(),
        SubmissionRetirement::CompleteSnapshot
    );
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(
        alias.try_metadata_snapshot().unwrap().allocation(),
        Some(allocation)
    );
    runtime_lock::try_retire(|| drop(alias)).unwrap();
    // Last native alias returns the physical root to the allocator cache.
    // Record retirement neither evicts that root nor runs Rust destructors.
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(
        try_retire_completed_submissions().unwrap(),
        SubmissionRetirement::CompleteSnapshot
    );
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    drop(_hook);
    crate::memory::clear_cache().unwrap();
    assert_eq!(retired.load(Ordering::SeqCst), 0);
    crate::reclaim_allocation_owners();
    assert_eq!(retired.load(Ordering::SeqCst), 1);
}

#[test]
fn retirement_pass_preserves_native_error_and_rejects_unknown_raw_result() {
    let error = runtime_lock::try_retire(|| {
        u32::try_from_op(|_| {
            // SAFETY: the C API explicitly rejects a null result pointer before
            // accessing it. Exercise the real native error channel, not a mock.
            unsafe { safemlx_sys::mlx_submission_retire_completed(std::ptr::null_mut()) }
        })
    })
    .unwrap()
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("[mlx_submission_retire_completed] Requires an output pointer."));
    assert!(error.to_string().contains("submission.cpp"));
    assert!(SubmissionRetirement::from_raw(u32::MAX).is_err());
    assert_eq!(
        try_retire_completed_submissions().unwrap(),
        SubmissionRetirement::CompleteSnapshot
    );
}

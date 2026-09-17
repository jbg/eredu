use super::*;
use crate::{
    PreparedHostTransferPlan, PreparedInputArena, PreparedInputRuntime,
    PreparedSubmissionGraphQuota, RetirementCapacity, utils::runtime_lock,
};

fn isolated(name: &str) -> bool {
    const KEY: &str = "SAFEMLX_EXCLUSIVE_HOST_WRITER_CASE";
    if std::env::var(KEY).ok().as_deref() == Some(name) {
        return true;
    }
    let module = module_path!().split_once("::").unwrap().1;
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("{module}::{name}"),
            "--nocapture",
            "--test-threads=1",
        ])
        .env(KEY, name)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed; 0 failed;"),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    false
}
fn writer() -> (PreparedHostTransferWriter, RetirementCapacity) {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let plan = PreparedHostTransferPlan::new(&runtime, &[4], crate::Dtype::Uint8, 0).unwrap();
    let capacity = RetirementCapacity::new(plan.backing_bytes());
    let permit = capacity.try_acquire(plan.backing_bytes()).unwrap();
    let quota =
        PreparedSubmissionGraphQuota::try_new_with_retirement(plan.metadata_bytes(), (), permit)
            .unwrap();
    let arena = PreparedInputArena::try_allocate(quota).unwrap();
    let writer = plan.construct_writer(&arena).unwrap();
    drop(arena);
    (writer, capacity)
}
#[test]
fn worker_fills_during_native_lock_and_returns_same_final_backing() {
    if !isolated("worker_fills_during_native_lock_and_returns_same_final_backing") {
        return;
    }
    let (writer, capacity) = writer();
    let occupied = capacity.occupied_bytes();
    assert!(occupied >= 4);
    // Holding the actual native lock makes an accidental worker try-lock fail
    // and a blocking native lock deadlock; byte filling needs neither.
    let guard = runtime_lock::enter();
    let worker = thread::spawn(move || {
        let mut writer = writer.try_return().expect_err("worker cannot publish");
        writer.as_bytes_mut().copy_from_slice(&[13, 21, 34, 55]);
        writer
    });
    let writer = worker.join().unwrap();
    drop(guard);
    let buffer = writer.try_return().unwrap().freeze();
    assert_eq!(buffer.as_bytes().unwrap(), &[13, 21, 34, 55]);
    assert_eq!(capacity.occupied_bytes(), occupied);
    drop(buffer);
    crate::reclaim_allocation_owners();
    assert_eq!(capacity.occupied_bytes(), 0);
}
#[test]
fn worker_unwind_keeps_source_capacity_until_host_retirement() {
    if !isolated("worker_unwind_keeps_source_capacity_until_host_retirement") {
        return;
    }
    let (mut writer, capacity) = writer();
    let occupied = capacity.occupied_bytes();
    let guard = runtime_lock::enter();
    let worker = thread::spawn(move || {
        writer.as_bytes_mut()[0] = 89;
        panic!("partial checkpoint read");
    });
    assert!(worker.join().is_err());
    assert_eq!(capacity.occupied_bytes(), occupied);
    assert!(capacity.try_acquire(1).is_err());
    drop(guard);
    // First pass retires the queued native buffer, which may itself queue the
    // arena's neutral custody. Capacity cannot return before actual release.
    crate::reclaim_allocation_owners();
    crate::reclaim_allocation_owners();
    assert_eq!(capacity.occupied_bytes(), 0);
}

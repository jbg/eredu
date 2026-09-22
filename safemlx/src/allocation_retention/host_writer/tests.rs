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

#[cfg(not(feature = "cuda"))]
#[test]
fn ordinary_writer_keeps_exclusive_bytes_and_creator_thread_retirement() {
    if !isolated("ordinary_writer_keeps_exclusive_bytes_and_creator_thread_retirement") {
        return;
    }
    let kind = if cfg!(feature = "metal") {
        crate::HostTransferStorageKind::MetalShared
    } else {
        crate::HostTransferStorageKind::Cpu
    };
    let initial = crate::host_transfer_memory_stats(kind).unwrap();
    let writer = PreparedHostTransferWriter::ordinary(&[4], crate::Dtype::Uint8).unwrap();
    let guard = runtime_lock::enter();
    let writer = thread::spawn(move || {
        let mut writer = writer.try_return().expect_err("worker cannot publish");
        writer.as_bytes_mut().copy_from_slice(&[3, 17, 43, 97]);
        writer
    })
    .join()
    .unwrap();
    drop(guard);
    let buffer = writer.try_return().unwrap().freeze();
    assert_eq!(buffer.as_bytes().unwrap(), &[3, 17, 43, 97]);
    let active = crate::host_transfer_memory_stats(kind).unwrap();
    assert_eq!(active.active_allocations, initial.active_allocations + 1);
    assert!(active.active_bytes > initial.active_bytes);
    drop(buffer);
    let mut writer = PreparedHostTransferWriter::ordinary(&[4], crate::Dtype::Uint8).unwrap();
    let guard = runtime_lock::enter();
    assert!(
        thread::spawn(move || {
            writer.as_bytes_mut()[0] = 101;
            panic!("partial ordinary Disk read");
        })
        .join()
        .is_err()
    );
    drop(guard);
    crate::reclaim_allocation_owners();
    crate::reclaim_allocation_owners();
    let retired = crate::host_transfer_memory_stats(kind).unwrap();
    assert_eq!(retired.active_allocations, initial.active_allocations);
    assert_eq!(retired.active_bytes, initial.active_bytes);
}

#[cfg(not(feature = "cuda"))]
#[test]
fn ordinary_writer_keeps_prepaid_owner_through_worker_unwind_and_aliases() {
    if !isolated("ordinary_writer_keeps_prepaid_owner_through_worker_unwind_and_aliases") {
        return;
    }
    let capacity = RetirementCapacity::new(1);
    let mut owner =
        Some(crate::PreparedAllocationOwner::try_new(capacity.try_acquire(1).unwrap()).unwrap());
    let writer = PreparedHostTransferWriter::ordinary_with_prepared_owner(
        &[4],
        crate::Dtype::Uint8,
        &mut owner,
    )
    .unwrap();
    assert!(owner.is_none());
    let guard = runtime_lock::enter();
    let writer = thread::spawn(move || {
        let mut writer = writer;
        writer.as_bytes_mut().copy_from_slice(&[7, 19, 53, 113]);
        writer
    })
    .join()
    .unwrap();
    assert_eq!(capacity.occupied_bytes(), 1);
    drop(guard);
    let buffer = std::sync::Arc::new(writer.try_return().unwrap().freeze());
    assert_eq!(buffer.as_bytes().unwrap(), &[7, 19, 53, 113]);
    let alias = buffer.clone();
    drop(buffer);
    crate::reclaim_allocation_owners();
    assert_eq!(capacity.occupied_bytes(), 1);
    drop(alias);
    crate::reclaim_allocation_owners();
    assert_eq!(capacity.occupied_bytes(), 0);

    let mut owner =
        Some(crate::PreparedAllocationOwner::try_new(capacity.try_acquire(1).unwrap()).unwrap());
    assert!(
        PreparedHostTransferWriter::ordinary_with_prepared_owner(
            &[-1],
            crate::Dtype::Uint8,
            &mut owner,
        )
        .is_err()
    );
    assert!(owner.is_some());
    assert_eq!(capacity.occupied_bytes(), 1);
    let mut writer = PreparedHostTransferWriter::ordinary_with_prepared_owner(
        &[4],
        crate::Dtype::Uint8,
        &mut owner,
    )
    .unwrap();
    let guard = runtime_lock::enter();
    assert!(
        thread::spawn(move || {
            writer.as_bytes_mut()[0] = 127;
            panic!("partial paid ordinary read");
        })
        .join()
        .is_err()
    );
    assert_eq!(capacity.occupied_bytes(), 1);
    assert!(capacity.try_acquire(1).is_err());
    drop(guard);
    crate::reclaim_allocation_owners();
    crate::reclaim_allocation_owners();
    assert_eq!(capacity.occupied_bytes(), 0);
}

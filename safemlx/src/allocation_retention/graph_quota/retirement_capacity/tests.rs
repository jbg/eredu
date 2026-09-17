use super::*;
use crate::{reclaim_allocation_owners, PreparedSubmissionScopeOwner, SubmissionScope};
use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

struct Owner(Arc<AtomicUsize>);
impl Drop for Owner {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn allocate<T: Send + 'static>(
    mut prepared: PreparedSubmissionGraphQuota<T>,
) -> SubmissionGraphQuota {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match prepared.try_allocate() {
            Ok(value) => return value,
            Err(error) => {
                assert_eq!(error.cause(), SubmissionGraphQuotaCause::RuntimeBusy);
                prepared = error.into_parts().1;
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
        }
    }
}

// These assertions deliberately observe the interval before arbitrary queued
// owners are reclaimed. Other tests may run the process-wide reclaimer, so run
// the two queue/physical-lifetime cases alone in this same test binary.
fn isolated_queue_case(name: &str) -> bool {
    const CHILD: &str = "SAFEMLX_RETIREMENT_CAPACITY_TEST_CHILD";
    if std::env::var(CHILD).ok().as_deref() == Some(name) {
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
        .env(CHILD, name)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.contains("1 passed; 0 failed;"),
        "isolated capacity case failed or did not select exactly one test:\n{stdout}\n{stderr}"
    );
    false
}

#[test]
fn capacity_cancellation_and_rejected_native_constructor_refund_once() {
    let capacity = RetirementCapacity::new(17);
    let count = Arc::new(AtomicUsize::new(0));
    let mut permit = capacity.try_acquire(17).unwrap();
    let split = permit.try_split(7).unwrap();
    assert_eq!(capacity.occupied_bytes(), 17);
    assert_eq!(
        permit.try_split(11).unwrap_err(),
        RetirementCapacityCause::Exhausted {
            requested: 11,
            available: 10
        }
    );
    assert_eq!(
        capacity.try_acquire(1).unwrap_err(),
        RetirementCapacityCause::Exhausted {
            requested: 1,
            available: 0
        }
    );
    let error =
        PreparedSubmissionGraphQuota::try_new_with_retirement(1, Owner(count.clone()), permit)
            .unwrap_err();
    assert_eq!(error.cause(), SubmissionGraphQuotaCause::InvalidCapacity);
    assert_eq!(capacity.occupied_bytes(), 17);
    drop(error);
    assert_eq!(capacity.occupied_bytes(), 7);
    assert_eq!(count.load(Ordering::SeqCst), 1);
    drop(split);
    assert_eq!(capacity.occupied_bytes(), 0);

    let mut prepared = PreparedSubmissionGraphQuota::try_new_with_retirement(
        64 * 1024,
        Owner(count.clone()),
        capacity.try_acquire(17).unwrap(),
    )
    .unwrap();
    // Exercise the real C constructor's refusal contract without an enormous
    // allocation or an allocator fault hook. Only its requested capacity is
    // invalidated; the exact prepared owner/node remains attached to the error.
    let identity = std::ptr::from_ref(&**prepared.node.as_ref().unwrap()) as usize;
    prepared.layout.capacity = 1;
    let deadline = Instant::now() + Duration::from_secs(5);
    let error = loop {
        match prepared.try_allocate() {
            Ok(_) => panic!("invalid native quota was accepted"),
            Err(error) if error.cause() == SubmissionGraphQuotaCause::RuntimeBusy => {
                prepared = error.into_parts().1;
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
            Err(error) => break error,
        }
    };
    assert_eq!(
        error.cause(),
        SubmissionGraphQuotaCause::NativeAllocationFailed
    );
    assert_eq!(
        std::ptr::from_ref(&**error.owner().node.as_ref().unwrap()) as usize,
        identity
    );
    assert_eq!(capacity.occupied_bytes(), 17);
    let returned = error.into_parts().1.into_owner().into_owner();
    assert_eq!(capacity.occupied_bytes(), 0);
    assert_eq!(count.load(Ordering::SeqCst), 1);
    drop(returned);
    assert_eq!(count.load(Ordering::SeqCst), 2);
    reclaim_allocation_owners();
    assert_eq!(capacity.occupied_bytes(), 0);
    assert_eq!(count.load(Ordering::SeqCst), 2);
}

#[test]
fn capacity_refunds_at_final_native_alias_before_arbitrary_owner_retirement() {
    if !isolated_queue_case(
        "capacity_refunds_at_final_native_alias_before_arbitrary_owner_retirement",
    ) {
        return;
    }
    let capacity = RetirementCapacity::new(23);
    let count = Arc::new(AtomicUsize::new(0));
    let prepared = PreparedSubmissionGraphQuota::try_new_with_retirement(
        64 * 1024,
        Owner(count.clone()),
        capacity.try_acquire(23).unwrap(),
    )
    .unwrap();
    let identity = std::ptr::from_ref(&**prepared.node.as_ref().unwrap()) as usize;
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        ready_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
    });
    let entered = ready_rx.recv_timeout(Duration::from_secs(5)).is_ok();
    let result = prepared.try_allocate();
    let released = release_tx.send(()).is_ok();
    let held = worker.join().unwrap();
    assert!(entered && released && held);
    let error = result.unwrap_err();
    assert_eq!(error.cause(), SubmissionGraphQuotaCause::RuntimeBusy);
    assert_eq!(
        std::ptr::from_ref(&**error.owner().node.as_ref().unwrap()) as usize,
        identity
    );
    assert_eq!(capacity.occupied_bytes(), 23);
    let quota = allocate(error.into_parts().1);
    let alias = quota.clone();
    let mut prepared_scope = PreparedSubmissionScopeOwner::try_new(())
        .unwrap()
        .with_graph_quota(quota.clone());
    let deadline = Instant::now() + Duration::from_secs(5);
    let scope = loop {
        match SubmissionScope::try_begin_retaining(prepared_scope) {
            Ok(value) => break value,
            Err(error) => {
                assert_eq!(error.cause(), crate::SubmissionScopeOwnerCause::RuntimeBusy);
                prepared_scope = error.into_parts().1;
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
        }
    };
    drop((quota, alias));
    assert_eq!(capacity.occupied_bytes(), 23);
    drop(scope);
    // Final native release refunds the bytes before arbitrary T destruction.
    assert_eq!(capacity.occupied_bytes(), 0);
    assert_eq!(count.load(Ordering::SeqCst), 0);
    let next = capacity.try_acquire(23).unwrap();
    reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert_eq!(capacity.occupied_bytes(), 23);
    drop(next);
    assert_eq!(capacity.occupied_bytes(), 0);
}

#[cfg(all(target_vendor = "apple", feature = "metal"))]
#[test]
fn host_source_backing_recycles_before_queued_custody_without_double_refund() {
    if !isolated_queue_case(
        "host_source_backing_recycles_before_queued_custody_without_double_refund",
    ) {
        return;
    }
    use crate::{
        host_transfer_memory_stats, Dtype, HostTransferStorageKind, PreparedHostTransferPlan,
        PreparedInputArena, PreparedInputRuntime,
    };
    let runtime = PreparedInputRuntime::prepare().unwrap();
    // Serialize native construction and observations, and prevent this test's
    // ordinary entries from reclaiming queued arbitrary owners during reuse.
    let guard = runtime_lock::enter();
    let baseline = host_transfer_memory_stats(HostTransferStorageKind::MetalShared).unwrap();
    let shape = [4];
    let values = [0.25f32, -2.0, 7.5, 9.0];
    let bytes = PreparedHostTransferPlan::new(&runtime, &shape, Dtype::Float32, 1)
        .unwrap()
        .backing_bytes();
    let capacity = RetirementCapacity::new(bytes);
    let count = Arc::new(AtomicUsize::new(0));
    let create = || {
        let plan = PreparedHostTransferPlan::new(&runtime, &shape, Dtype::Float32, 1).unwrap();
        let permit = capacity.try_acquire(plan.backing_bytes()).unwrap();
        let prepared = PreparedSubmissionGraphQuota::try_new_with_retirement(
            plan.metadata_bytes(),
            Owner(count.clone()),
            permit,
        )
        .unwrap();
        let arena = PreparedInputArena::try_allocate(prepared).unwrap();
        let mut source = plan.construct(&arena).unwrap();
        for (destination, value) in source
            .as_bytes_mut()
            .unwrap()
            .chunks_exact_mut(4)
            .zip(values)
        {
            destination.copy_from_slice(&value.to_ne_bytes());
        }
        (source.freeze(), arena)
    };
    let (source, arena) = create();
    let alias = source.try_prepared_source_array().unwrap();
    assert_eq!(alias.evaluated().unwrap().try_as_slice::<f32>().unwrap(), &values);
    let first = host_transfer_memory_stats(HostTransferStorageKind::MetalShared).unwrap();
    assert_eq!(first.active_bytes, baseline.active_bytes + bytes);
    assert_eq!(first.active_allocations, baseline.active_allocations + 1);
    assert!(matches!(
        capacity.try_acquire(bytes),
        Err(RetirementCapacityCause::Exhausted { .. })
    ));
    let refused = host_transfer_memory_stats(HostTransferStorageKind::MetalShared).unwrap();
    assert_eq!(
        (refused.active_bytes, refused.active_allocations),
        (first.active_bytes, first.active_allocations)
    );
    drop((source, arena));
    assert_eq!(capacity.occupied_bytes(), bytes);
    assert_eq!(alias.evaluated().unwrap().try_as_slice::<f32>().unwrap(), &values);
    drop(alias);
    assert_eq!(capacity.occupied_bytes(), 0);
    assert_eq!(count.load(Ordering::SeqCst), 0);
    let retired = host_transfer_memory_stats(HostTransferStorageKind::MetalShared).unwrap();
    assert_eq!(
        (retired.active_bytes, retired.active_allocations),
        (baseline.active_bytes, baseline.active_allocations)
    );

    // Reuse actual native backing while the preceding request owner is queued.
    let (next, next_arena) = create();
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert_eq!(capacity.occupied_bytes(), bytes);
    drop(guard);
    reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 1);
    assert_eq!(capacity.occupied_bytes(), bytes);
    drop((next, next_arena));
    assert_eq!(capacity.occupied_bytes(), 0);
    reclaim_allocation_owners();
    assert_eq!(count.load(Ordering::SeqCst), 2);
    assert_eq!(capacity.occupied_bytes(), 0);
}

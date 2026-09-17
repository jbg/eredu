use super::*;
use crate::{
    reclaim_allocation_owners, transforms::async_eval_with_event, Array, Device, DeviceType,
    Stream, SubmissionActivity, SubmissionRecordActivity,
};
use std::{
    cell::Cell,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc, Mutex,
    },
    time::{Duration, Instant},
};

struct Payload {
    values: [u32; 4],
    drops: Arc<AtomicUsize>,
    thread: Arc<Mutex<Option<std::thread::ThreadId>>>,
}
impl Drop for Payload {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            assert!(crate::can_reclaim_submission_resources());
            assert_eq!(self.values, [3, 5, 7, 11]);
        }
        *self.thread.lock().unwrap() = Some(std::thread::current().id());
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
fn payload() -> (
    Payload,
    Arc<AtomicUsize>,
    Arc<Mutex<Option<std::thread::ThreadId>>>,
) {
    let drops = Arc::new(AtomicUsize::new(0));
    let thread = Arc::new(Mutex::new(None));
    (
        Payload {
            values: [3, 5, 7, 11],
            drops: drops.clone(),
            thread: thread.clone(),
        },
        drops,
        thread,
    )
}
fn begin<T: Send + 'static>(mut owner: PreparedSubmissionScopeOwner<T>) -> SubmissionScope {
    let end = Instant::now() + Duration::from_secs(5);
    loop {
        match SubmissionScope::try_begin_retaining(owner) {
            Ok(scope) => return scope,
            Err(error) => {
                assert_eq!(error.cause(), SubmissionScopeOwnerCause::RuntimeBusy);
                owner = error.into_parts().1;
                assert!(
                    Instant::now() < end,
                    "unrelated runtime contention did not end"
                );
                std::thread::yield_now();
            }
        }
    }
}
thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.with(|n| n.set(n.get() + 1));
}
struct Hook;
impl Hook {
    fn new() -> Self {
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

#[test]
fn allocation_failure_returns_exact_original_and_unattached_owner_drops_unlocked() {
    let (value, drops, _) = payload();
    let error = PreparedSubmissionScopeOwner::try_new_with(value, |_| ptr::null_mut()).unwrap_err();
    assert_eq!(error.cause(), SubmissionScopeOwnerCause::AllocationFailed);
    assert!(Arc::ptr_eq(&error.owner().drops, &drops));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let prepared = PreparedSubmissionScopeOwner::try_new(error.into_parts().1).unwrap();
    assert_eq!(prepared.owner().values, [3, 5, 7, 11]);
    drop(prepared);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn busy_keeps_same_preparation_without_callbacks_and_retry_is_constructor_only() {
    let (value, drops, _) = payload();
    let prepared = PreparedSubmissionScopeOwner::try_new(value).unwrap();
    let mut outer = SubmissionScope::begin().unwrap();
    let before = outer.status();
    let _hook = Hook::new();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _guard = runtime_lock::enter();
        entered_tx.send(()).unwrap();
        release_rx.recv_timeout(Duration::from_secs(5)).is_ok()
    });
    let entered = entered_rx.recv_timeout(Duration::from_secs(5)).is_ok();
    let result = SubmissionScope::try_begin_retaining(prepared);
    let released = release_tx.send(()).is_ok();
    let held = worker.join().unwrap();
    assert!(
        entered && released && held,
        "try_begin_retaining waited for the runtime"
    );
    let error = result.unwrap_err();
    assert_eq!(error.cause(), SubmissionScopeOwnerCause::RuntimeBusy);
    assert!(Arc::ptr_eq(&error.owner().owner().drops, &drops));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(outer.status(), before);
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    let child = begin(error.into_parts().1);
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    drop(child);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    reclaim_allocation_owners();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    outer.seal();
}

#[test]
fn native_failure_returns_same_capsule_after_runtime_loan() {
    unsafe extern "C" fn reject(
        _: *mut safemlx_sys::mlx_submission_scope,
        _: *mut c_void,
        _: Option<unsafe extern "C" fn(*mut c_void)>,
    ) -> i32 {
        1
    }
    let (value, drops, _) = payload();
    let mut prepared = PreparedSubmissionScopeOwner::try_new(value).unwrap();
    let end = Instant::now() + Duration::from_secs(5);
    let error = loop {
        let error = prepared.begin_with(reject).unwrap_err();
        if error.cause() != SubmissionScopeOwnerCause::RuntimeBusy {
            break error;
        }
        prepared = error.into_parts().1;
        assert!(Instant::now() < end);
        std::thread::yield_now();
    };
    assert_eq!(
        error.cause(),
        SubmissionScopeOwnerCause::NativeConstructionFailed
    );
    assert!(crate::can_reclaim_submission_resources());
    assert!(Arc::ptr_eq(&error.owner().owner().drops, &drops));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn quiescent_sealed_child_still_retains_durable_owned_parent_and_host_queue() {
    let (value, drops, thread) = payload();
    let _hook = Hook::new();
    let layout = PreparedSubmissionScopeOwner::<Payload>::layout().unwrap();
    assert!(layout.allocation_bytes().unwrap() > 0);
    let mut parent = begin(PreparedSubmissionScopeOwner::try_new(value).unwrap());
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    let mut child = SubmissionScope::begin().unwrap();
    HOUSEKEEPING.with(|n| n.set(0));
    parent.seal();
    assert_eq!(
        parent.accepted_records().activity(),
        SubmissionRecordActivity::Quiescent
    );
    assert_eq!(parent.status().activity(), SubmissionActivity::Pending);
    child.seal();
    assert_eq!(parent.status().activity(), SubmissionActivity::None);
    drop(parent);
    reclaim_allocation_owners();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(child);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    let reclaimer = std::thread::spawn(|| {
        reclaim_allocation_owners();
        std::thread::current().id()
    })
    .join()
    .unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(*thread.lock().unwrap(), Some(reclaimer));
}

#[test]
fn actual_nonzero_cpu_work_uses_owned_scope_without_readiness_releasing_payload() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let input = Array::from_slice(&[2_f32, -3., 5., 7.], &[4]);
    let (value, drops, _) = payload();
    let mut scope = begin(PreparedSubmissionScopeOwner::try_new(value).unwrap());
    let output = input.square(&stream).unwrap();
    let event = async_eval_with_event([&output]).unwrap();
    scope.seal();
    event.synchronize().unwrap();
    assert_eq!(
        output.evaluated().unwrap().as_slice::<f32>(),
        &[4., 9., 25., 49.]
    );
    let status = scope.progress();
    assert!(status.is_settled() && !status.failed() && !status.blocked());
    assert_eq!(
        scope.accepted_records().activity(),
        SubmissionRecordActivity::Quiescent
    );
    reclaim_allocation_owners();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(event);
    drop(output);
    drop(input);
    drop(scope);
    let end = Instant::now() + Duration::from_secs(5);
    while drops.load(Ordering::SeqCst) == 0 && Instant::now() < end {
        crate::try_retire_completed_submissions().unwrap();
        reclaim_allocation_owners();
        std::thread::yield_now();
    }
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn scope_unwind_seals_and_queues_without_running_owner_destructor() {
    let mut outer = SubmissionScope::begin().unwrap();
    let (value, drops, _) = payload();
    let sentinel = Arc::new(37usize);
    let expected = sentinel.clone();
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _scope = begin(PreparedSubmissionScopeOwner::try_new(value).unwrap());
        std::panic::panic_any(sentinel);
    }))
    .unwrap_err();
    assert!(Arc::ptr_eq(
        panic.downcast_ref::<Arc<usize>>().unwrap(),
        &expected
    ));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    // The real outer scope no longer has a structural pending child. A fresh
    // scope attaches to that same live outer, proving restored TLS ancestry.
    assert_eq!(outer.status().activity(), SubmissionActivity::None);
    let scope = SubmissionScope::try_begin().unwrap();
    assert_eq!(outer.status().activity(), SubmissionActivity::Pending);
    assert_eq!(scope.status().activity(), SubmissionActivity::None);
    drop(scope);
    assert_eq!(outer.status().activity(), SubmissionActivity::None);
    outer.seal();
    reclaim_allocation_owners();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

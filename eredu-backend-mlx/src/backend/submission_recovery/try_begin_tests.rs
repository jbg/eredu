use super::*;
use eredu_core::*;
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, InferenceRequest, WorkingMemoryPool,
};
use safemlx::{transforms::async_eval_with_event, Array, Device, DeviceType, Stream};
use std::{
    cell::Cell,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
    time::Duration,
};

// A charge-lifetime fixture only, not a numerical bound for the native arrays.
fn request() -> (WorkingMemoryPool, InferenceRequest, u64) {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 3,
        max_output_tokens: 1,
        prefill_chunk_positions: 2,
        output: OutputDemand::LastPosition,
    };
    let request = AdmissionRequest {
        input: InputTokenCount::text(3),
        max_output_tokens: 1,
        batch_size: 1,
        safety_reserve_bytes: 0,
        application_memory_budget_bytes: None,
        require_complete_estimate: true,
    };
    let capabilities = ModelCapabilities {
        effective_model_type: "try-begin custody fixture".into(),
        native_max_context: Observed::exact(8, "fixture"),
        effective_max_context: Observed::exact(8, "fixture"),
        state_strategy: CacheStateStrategy::FullKv,
        modalities: InputModalities::TEXT,
        estimation: EstimationCompleteness::PersistentStateOnly,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::empty(),
        vec![],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let bound = || WorkspaceBound::bounded(16, "request custody fixture");
    let state = estimate_runtime_state(
        &layout,
        request.input,
        1,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(ExecutionWorkspaceEstimate {
        geometry,
        activations: bound(),
        attention: bound(),
        vocabulary: bound(),
        state_update: bound(),
        materialization: bound(),
        retained: bound(),
    })
    .unwrap();
    let AdmissionResult::Admitted(admission) =
        apply_admission_policy(&capabilities, request, state, None).unwrap()
    else {
        panic!("fixture admission")
    };
    let charge = admission.incremental_required_bytes;
    let pool = WorkingMemoryPool::new(charge, 0).unwrap();
    let reservation = pool
        .reserve(&InferenceExecutionIdentity::default(), &admission)
        .unwrap();
    (pool, reservation.into(), charge)
}
struct Resources {
    arrays: Vec<Array>,
    drops: Arc<AtomicUsize>,
    request: InferenceRequest,
}
impl Retention for Resources {
    fn observe(&self, _: Status) {}
}
impl Drop for Resources {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}
fn resources(request: InferenceRequest, drops: &Arc<AtomicUsize>) -> Resources {
    let mut arrays = Vec::with_capacity(2);
    arrays.push(Array::from_slice(&[2_f32, -3., 5., 7.], &[4]));
    Resources {
        arrays,
        drops: drops.clone(),
        request,
    }
}
fn with_foreign_runtime<T>(operation: impl FnOnce() -> T) -> T {
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
    let result = operation();
    let released = release_tx.send(()).is_ok();
    let held_until_release = worker.join().unwrap();
    assert!(
        released && held_until_release,
        "operation waited for the foreign runtime"
    );
    result
}
thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.with(|n| n.set(n.get() + 1));
}
struct Hook;
impl Hook {
    fn install() -> Self {
        safemlx::register_thread_runtime_housekeeping(housekeeping);
        HOUSEKEEPING.with(|n| n.set(0));
        Self
    }
}
impl Drop for Hook {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

#[test]
fn busy_returns_the_exact_resource_owner_and_request_without_copy_or_reap() {
    reap();
    let (pool, request, charge) = request();
    let drops = Arc::new(AtomicUsize::new(0));
    let resources = resources(request, &drops);
    let request_identity = resources.request.clone();
    let ptr = resources.arrays.as_ptr();
    let capacity = resources.arrays.capacity();
    let allocation = resources.arrays[0]
        .try_metadata_snapshot()
        .unwrap()
        .allocation();
    let _hook = Hook::install();
    let error = with_foreign_runtime(|| Recovery::try_begin(resources)).unwrap_err();
    assert!(matches!(error.cause(), SubmissionScopeBeginError::Busy));
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(pool.used_bytes().unwrap(), charge);
    assert_eq!(error.retention().arrays.as_ptr(), ptr);
    error
        .retention()
        .request
        .validate_same_request(&request_identity)
        .unwrap();
    drop(request_identity);
    let (resources, cause) = error.into_parts();
    assert!(matches!(cause, SubmissionScopeBeginError::Busy));
    assert_eq!(resources.arrays.as_ptr(), ptr);
    assert_eq!(resources.arrays.capacity(), capacity);
    assert_eq!(
        resources.arrays[0]
            .try_metadata_snapshot()
            .unwrap()
            .allocation(),
        allocation
    );
    assert_eq!(resources.request.geometry().input_positions, 3);
    drop(resources);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn dropping_busy_error_releases_only_its_unused_owner() {
    let (pool, request, charge) = request();
    let drops = Arc::new(AtomicUsize::new(0));
    let resources = resources(request, &drops);
    let alias = resources.arrays[0].clone();
    let allocation = alias.try_metadata_snapshot().unwrap().allocation();
    let error = with_foreign_runtime(|| Recovery::try_begin(resources)).unwrap_err();
    assert_eq!(pool.used_bytes().unwrap(), charge);
    assert!(std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<SubmissionScopeBeginError>()
        .is_some());
    drop(error);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(
        alias.try_metadata_snapshot().unwrap().allocation(),
        allocation
    );
    assert_eq!(
        alias.evaluated().unwrap().as_slice::<f32>(),
        &[2., -3., 5., 7.]
    );
}

#[test]
fn successful_try_begin_keeps_resources_until_explicit_reap_after_real_completion() {
    reap();
    // This test owns its thread's service registration. A try-only user must
    // arrange that service separately; try_begin must not silently install it.
    safemlx::unregister_thread_runtime_housekeeping(reap);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let (pool, request, charge) = request();
    let drops = Arc::new(AtomicUsize::new(0));
    let resources = resources(request, &drops);
    let ptr = resources.arrays.as_ptr();
    let _hook = Hook::install();
    let mut recovery = Recovery::try_begin(resources).unwrap();
    assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
    assert_eq!(recovery.retention().arrays.as_ptr(), ptr);
    let output = recovery.retention().arrays[0].square(&stream).unwrap();
    recovery.retention_mut().arrays.push(output.clone());
    let completion = async_eval_with_event([&output]).unwrap();
    recovery.seal();
    // Lock contention makes destruction unobservable even if CPU work already
    // finished. Actual event completion is observed only after recovery Drop.
    with_foreign_runtime(|| drop(recovery));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(pool.used_bytes().unwrap(), charge);
    completion.synchronize().unwrap();
    let before = HOUSEKEEPING.with(Cell::get);
    let ordinary = Array::from_slice(&[11_i32], &[1]);
    // from_slice alone does not enter the runtime housekeeping path.
    assert_eq!(ordinary.evaluated().unwrap().as_slice::<i32>(), &[11]);
    assert!(HOUSEKEEPING.with(Cell::get) > before);
    assert_eq!(
        drops.load(Ordering::SeqCst),
        0,
        "try_begin registered an automatic reaper"
    );
    assert_eq!(pool.used_bytes().unwrap(), charge);
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while drops.load(Ordering::SeqCst) == 0 {
        reap();
        assert!(
            std::time::Instant::now() < deadline,
            "exact completion did not retire owner"
        );
        std::thread::yield_now();
    }
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(
        output.evaluated().unwrap().as_slice::<f32>(),
        &[4., 9., 25., 49.]
    );
    drop(ordinary);
}

#[test]
fn nested_and_outer_abandon_pending_without_failure_or_blocked_completion() {
    struct PendingProbe {
        status: Rc<Cell<Status>>,
        polls: Rc<Cell<usize>>,
    }
    impl Probe for PendingProbe {
        fn seal(&mut self) {}
        fn progress(&self) -> Status {
            self.polls.set(self.polls.get() + 1);
            self.status.get()
        }
    }
    reap();
    let (pool, request, charge) = request();
    let drops = Arc::new(AtomicUsize::new(0));
    let status = Rc::new(Cell::new(Status {
        settled: false,
        failed: false,
        blocked: false,
    }));
    let polls = Rc::new(Cell::new(0));
    let nested = Recovery::with_probe(
        resources(request.clone(), &drops),
        PendingProbe {
            status: status.clone(),
            polls: polls.clone(),
        },
    );
    let outer = Recovery::with_probe(
        resources(request, &drops),
        PendingProbe {
            status: status.clone(),
            polls: polls.clone(),
        },
    );
    drop(nested);
    drop(outer);
    assert_eq!(
        polls.get(),
        2,
        "one bounded retirement attempt per owner, never Recovery::finish"
    );
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(pool.used_bytes().unwrap(), charge);
    // Only independent completion evidence permits release. A deadline alone did not.
    status.set(Status {
        settled: true,
        failed: false,
        blocked: false,
    });
    for _ in 0..8 {
        reap();
        if drops.load(Ordering::SeqCst) == 2 {
            break;
        }
    }
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn ordinary_prefill_busy_retires_only_unused_request_and_preserves_native_cause() {
    let _hook = Hook::install();
    // Unreserved ordinary completion creates a fresh thread-local empty root
    // owner. Reserved compatibility has no collector. Neither has submitted.
    for reserved in [false, true] {
        let (pool, paid, charge) = request();
        let request = if reserved {
            paid
        } else {
            let request = InferenceRequest::without_memory_budget(
                &InferenceExecutionIdentity::default(),
                paid.geometry(),
            )
            .unwrap();
            drop(paid);
            request
        };
        let identity = request.clone();
        let error = with_foreign_runtime(|| prefill::begin_ordinary(request)).unwrap_err();
        let crate::backend::error::Error::Other(owner) = &error else {
            panic!("ordinary prefill lost the supplied request")
        };
        let owner = owner
            .downcast_ref::<RecoveryBeginError<prefill::PrefillRequestRetention>>()
            .unwrap();
        owner
            .retention()
            .0
            .validate_same_request(&identity)
            .unwrap();
        assert!(matches!(
            std::error::Error::source(owner)
                .unwrap()
                .downcast_ref::<SubmissionScopeBeginError>(),
            Some(SubmissionScopeBeginError::Busy)
        ));
        assert_eq!(HOUSEKEEPING.with(Cell::get), 0);
        drop(identity);
        assert_eq!(
            pool.used_bytes().unwrap(),
            if reserved { charge } else { 0 }
        );
        // The error owns the request, but neither roots nor accepted work.
        // Last-owner retirement on another thread proves no Rc escaped.
        fn requires_send_sync<T: Send + Sync>(_: &T) {}
        requires_send_sync(&error);
        std::thread::spawn(move || drop(error)).join().unwrap();
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

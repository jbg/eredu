use super::*;
use eredu_nn::workspace::WorkspaceMetadataAccount;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc,
};
use std::time::Duration;

#[derive(Debug)]
struct State {
    remaining: Mutex<usize>,
    retired: AtomicBool,
    errors: AtomicUsize,
}
#[derive(Debug)]
struct Account(Arc<State>);
impl WorkspaceMetadataAccount for Account {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), eredu_core::HostMetadataFundingError> {
        let mut remaining = self.0.remaining.lock().unwrap();
        *remaining =
            remaining
                .checked_sub(bytes)
                .ok_or(eredu_core::HostMetadataFundingError::Capacity {
                    required: bytes as u64,
                    available: *remaining as u64,
                })?;
        Ok(())
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.0.retired.store(true, Ordering::SeqCst);
    }
}
#[derive(Debug)]
struct Failure {
    code: usize,
    source: Arc<State>,
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "failure {}", self.code)
    }
}
impl BackgroundPrefetchFailure for Failure {
    fn from_panic(_: Box<dyn std::any::Any + Send>) -> Self {
        panic!("test operation must not panic")
    }
}
impl Drop for Failure {
    fn drop(&mut self) {
        assert!(
            !self.source.retired.load(Ordering::SeqCst),
            "terminal source outlives failure retirement"
        );
        self.source
            .errors
            .fetch_or(1 << self.code, Ordering::SeqCst);
    }
}
fn account() -> (Arc<State>, WorkspaceMetadataFunding) {
    let state = Arc::new(State {
        remaining: Mutex::new(usize::MAX),
        retired: AtomicBool::new(false),
        errors: AtomicUsize::new(0),
    });
    let funding = WorkspaceMetadataFunding::new(Account(state.clone())).unwrap();
    (state, funding)
}
fn ids() -> Vec<OffloadUnitId> {
    ["a", "b", "c"]
        .into_iter()
        .map(|s| OffloadUnitId::new(s).unwrap())
        .collect()
}

#[test]
fn paid_selected_storage_refuses_before_birth_and_retains_escaped_source_failure() {
    let units = ids();
    let (state, funding) = account();
    let bytes = PreparedPrefetchStorage::<Failure>::metadata_bytes(&units, 2).unwrap();
    *state.remaining.lock().unwrap() = bytes - 1;
    let error =
        PreparedPrefetchStorage::<Failure>::prepare(&units, 2, funding.clone()).unwrap_err();
    assert!(matches!(
        error.cause,
        PreparationCause::Funding(eredu_core::HostMetadataFundingError::Capacity { .. })
    ));
    assert!(error.units.is_empty());
    assert_eq!(*state.remaining.lock().unwrap(), bytes - 1);
    drop(error);
    *state.remaining.lock().unwrap() = usize::MAX;
    let storage = PreparedPrefetchStorage::prepare(&units, 2, funding).unwrap();
    let operation_state = state.clone();
    let (pointer_tx, pointer_rx) = mpsc::channel();
    let worker = BackgroundPrefetchWorker::for_prepared_units_retaining(
        storage,
        "prepared-prefetch-source",
        move |id| {
            let code = id.as_str().as_bytes()[0] as usize - b'a' as usize;
            if code == 0 {
                pointer_tx.send(id.as_str().as_ptr() as usize).unwrap();
            }
            Err(Failure {
                code,
                source: operation_state.clone(),
            })
        },
    )
    .unwrap();
    for unit in &units {
        worker.submit(unit, false).unwrap();
    }
    worker.wait_idle().unwrap();
    *state.remaining.lock().unwrap() = worker.cancellation_metadata_bytes().unwrap() - 1;
    assert!(matches!(
        worker.cancel(),
        Err(BackgroundPrefetchWorkerError::HostMetadata(_))
    ));
    assert_eq!(worker.report().unwrap().failed(), 3);
    assert_eq!(state.errors.load(Ordering::SeqCst), 0);
    *state.remaining.lock().unwrap() = usize::MAX;
    let error = worker.cancel().unwrap_err();
    let BackgroundPrefetchWorkerError::SourceOperationFailed { unit, message } = &error else {
        panic!("source-backed failure")
    };
    assert_eq!(unit.id(), &units[0]);
    assert_eq!(message.code, 0);
    assert_eq!(
        unit.id().as_str().as_ptr() as usize,
        pointer_rx.recv().unwrap()
    );
    assert_eq!(state.errors.load(Ordering::SeqCst), (1 << 1) | (1 << 2));
    let escaped = unit.clone();
    drop(worker);
    drop(error);
    assert_eq!(state.errors.load(Ordering::SeqCst), 7);
    assert!(!state.retired.load(Ordering::SeqCst));
    drop(escaped);
    assert!(state.retired.load(Ordering::SeqCst));
}

#[test]
fn detached_worker_keeps_paid_queue_and_shutdown_failures_until_actual_return() {
    let (state, funding) = account();
    let units = ids();
    let storage = PreparedPrefetchStorage::prepare(&units, 2, funding).unwrap();
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let operation_gate = gate.clone();
    let operation_state = state.clone();
    let (started_tx, started_rx) = mpsc::channel();
    let worker = BackgroundPrefetchWorker::for_prepared_units_retaining(
        storage,
        "prepared-prefetch-detached",
        move |_| {
            started_tx.send(()).unwrap();
            let mut ready = operation_gate.0.lock().unwrap();
            while !*ready {
                ready = operation_gate.1.wait(ready).unwrap();
            }
            Err(Failure {
                code: 0,
                source: operation_state.clone(),
            })
        },
    )
    .unwrap()
    .with_nonblocking_drop();
    worker.submit(&units[0], false).unwrap();
    started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    worker.submit(&units[1], false).unwrap();
    drop(worker);
    assert!(!state.retired.load(Ordering::SeqCst));
    *gate.0.lock().unwrap() = true;
    gate.1.notify_all();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !state.retired.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "actual worker retirement");
        thread::yield_now();
    }
    assert_eq!(state.errors.load(Ordering::SeqCst), 1);
}

use super::*;
use std::{
    any::Any,
    error::Error,
    fmt,
    sync::{Weak, atomic::AtomicUsize},
};

type TypedLifecycle = storage::Funded<(Mutex<Shared<OwnedFailure>>, Condvar)>;

// Deliberately not Clone: demand and cancellation must move this owner.
#[derive(Debug)]
enum OwnedFailure {
    Cause {
        code: usize,
        lifecycle: Weak<TypedLifecycle>,
        dropped: Arc<AtomicUsize>,
        dropped_locked: Arc<AtomicBool>,
    },
    Panic(BackgroundPrefetchPanic),
}
impl OwnedFailure {
    fn code(&self) -> usize {
        match self {
            Self::Cause { code, .. } => *code,
            Self::Panic(_) => 0,
        }
    }
}
impl fmt::Display for OwnedFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cause { code, .. } => write!(f, "retained source failure {code}"),
            Self::Panic(panic) => fmt::Display::fmt(panic, f),
        }
    }
}
impl Error for OwnedFailure {}
impl BackgroundPrefetchFailure for OwnedFailure {
    fn from_panic(payload: Box<dyn Any + Send>) -> Self {
        Self::Panic(BackgroundPrefetchPanic::new(payload))
    }
    fn error_source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self)
    }
}
impl Drop for OwnedFailure {
    fn drop(&mut self) {
        if let Self::Cause {
            code,
            lifecycle,
            dropped,
            dropped_locked,
        } = self
        {
            let lifecycle = lifecycle
                .upgrade()
                .expect("worker remains alive during error retirement");
            if lifecycle.value.0.try_lock().is_err() {
                dropped_locked.store(true, Ordering::SeqCst);
            }
            dropped.fetch_or(1 << *code, Ordering::SeqCst);
        }
    }
}

#[test]
fn typed_failure_survives_demand_and_cancel_and_retires_outside_lock() {
    let units: Vec<_> = ["demand", "first", "second"]
        .into_iter()
        .map(|name| OffloadUnitId::new(name).unwrap())
        .collect();
    let lifecycle = Arc::new(Mutex::new(Weak::<TypedLifecycle>::new()));
    let dropped = Arc::new(AtomicUsize::new(0));
    let dropped_locked = Arc::new(AtomicBool::new(false));
    let operation_lifecycle = lifecycle.clone();
    let operation_dropped = dropped.clone();
    let operation_locked = dropped_locked.clone();
    let mut worker = BackgroundPrefetchWorker::for_units_retaining(
        3,
        units.clone(),
        "runtime-prefetch-typed-failure",
        move |unit| {
            let code = match unit.as_str() {
                "demand" => 1,
                "first" => 2,
                "second" => 3,
                _ => unreachable!(),
            };
            Err(OwnedFailure::Cause {
                code,
                lifecycle: operation_lifecycle.lock().unwrap().clone(),
                dropped: operation_dropped.clone(),
                dropped_locked: operation_locked.clone(),
            })
        },
    )
    .unwrap();
    *lifecycle.lock().unwrap() = Arc::downgrade(worker.shared.raw_for_test());
    for unit in &units {
        worker.submit(unit, false).unwrap();
    }
    worker.wait_idle().unwrap();
    // Stop the now-idle producer before the destructor probes: a failed
    // try_lock then proves consumer-held locking, not unrelated worker activity.
    worker.request_shutdown(false);
    worker.worker.take().unwrap().join().unwrap();
    assert_eq!(dropped.load(Ordering::SeqCst), 0);

    let PrefetchDemandResolution::Failed(cause) = worker.wait(&units[0]).unwrap() else {
        panic!("demand must receive its actual typed failure");
    };
    assert_eq!(cause.code(), 1);
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    drop(cause);
    assert_eq!(dropped.load(Ordering::SeqCst), 1 << 1);

    let error = worker.cancel().unwrap_err();
    let BackgroundPrefetchWorkerError::OperationFailed { id, message } = &error else {
        panic!("cancellation must return its first typed failure");
    };
    assert_eq!(id, &units[1]);
    assert_eq!(message.code(), 2);
    assert_eq!(
        error
            .source()
            .unwrap()
            .downcast_ref::<OwnedFailure>()
            .unwrap()
            .code(),
        2
    );
    // The later error has retired, while the returned owner is still retained.
    assert_eq!(dropped.load(Ordering::SeqCst), (1 << 1) | (1 << 3));
    assert!(!dropped_locked.load(Ordering::SeqCst));
    drop(error);
    assert_eq!(
        dropped.load(Ordering::SeqCst),
        (1 << 1) | (1 << 2) | (1 << 3)
    );
    assert!(!dropped_locked.load(Ordering::SeqCst));
    worker.cancel().unwrap();
}

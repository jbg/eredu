//! Backend-neutral bounded background weight prefetch execution.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Condvar, Mutex,
    },
    thread::{self, JoinHandle},
    time::Instant,
};

use eredu_core::residency::{
    BackgroundPrefetchReport, OffloadUnitId, PrefetchAdmission, PrefetchDemandResolution,
    PrefetchExecutionState,
};

enum WorkerMessage {
    WorkAvailable,
    Shutdown,
}

type PrefetchOperation = Arc<dyn Fn(&OffloadUnitId) -> Result<(), String> + Send + Sync + 'static>;

/// One bounded background prefetch worker with exact cancellation and configurable shutdown.
pub struct BackgroundPrefetchWorker {
    sender: mpsc::Sender<WorkerMessage>,
    shared: Arc<(Mutex<PrefetchExecutionState<String>>, Condvar)>,
    worker: Option<JoinHandle<()>>,
    nonblocking_drop: bool,
    stopping: Arc<AtomicBool>,
}

impl BackgroundPrefetchWorker {
    /// Starts a named worker which executes one backend-owned operation at a time.
    pub fn new<F>(
        capacity: usize,
        thread_name: impl Into<String>,
        operation: F,
    ) -> Result<Self, BackgroundPrefetchWorkerError>
    where
        F: Fn(&OffloadUnitId) -> Result<(), String> + Send + Sync + 'static,
    {
        let operation: PrefetchOperation = Arc::new(operation);
        let (sender, receiver) = mpsc::channel();
        let shared = Arc::new((
            Mutex::new(PrefetchExecutionState::new(capacity)?),
            Condvar::new(),
        ));
        let worker_shared = Arc::clone(&shared);
        let stopping = Arc::new(AtomicBool::new(false));
        let worker_stopping = Arc::clone(&stopping);
        let worker = thread::Builder::new()
            .name(thread_name.into())
            .spawn(move || worker_loop(operation, receiver, worker_shared, worker_stopping))?;
        Ok(Self {
            sender,
            shared,
            worker: Some(worker),
            nonblocking_drop: false,
            stopping,
        })
    }

    /// Requests shutdown without joining when this handle is dropped.
    ///
    /// The worker retains its operation and lifecycle state until in-flight
    /// work returns. Queued work is cancelled by that worker; callers needing
    /// a synchronous fence can still explicitly call [`Self::cancel`].
    pub fn with_nonblocking_drop(mut self) -> Self {
        self.nonblocking_drop = true;
        self
    }

    /// Admits or coalesces one operation after the backend reports current residency.
    pub fn submit(
        &self,
        id: &OffloadUnitId,
        resident: bool,
    ) -> Result<(), BackgroundPrefetchWorkerError> {
        let mut backpressure_started: Option<Instant> = None;
        loop {
            let mut state = self
                .shared
                .0
                .lock()
                .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?;
            match state.admit(id.clone(), resident) {
                PrefetchAdmission::Coalesced => {
                    if let Some(started) = backpressure_started {
                        state.finish_backpressure(started.elapsed());
                    }
                    return Ok(());
                }
                PrefetchAdmission::AtCapacity => {
                    if backpressure_started.is_none() {
                        state.begin_backpressure();
                        backpressure_started = Some(Instant::now());
                    }
                    drop(
                        self.shared
                            .1
                            .wait(state)
                            .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?,
                    );
                }
                PrefetchAdmission::Admitted(work) => {
                    if let Some(started) = backpressure_started {
                        state.finish_backpressure(started.elapsed());
                    }
                    if self.sender.send(WorkerMessage::WorkAvailable).is_ok() {
                        return Ok(());
                    }
                    state.rollback_admission(&work)?;
                    self.shared.1.notify_all();
                    return Err(BackgroundPrefetchWorkerError::WorkerDisconnected);
                }
            }
        }
    }

    /// Waits for background ownership to resolve and consumes its exact result.
    pub fn wait(
        &self,
        id: &OffloadUnitId,
    ) -> Result<PrefetchDemandResolution<String>, BackgroundPrefetchWorkerError> {
        let started = Instant::now();
        let mut state = self
            .shared
            .0
            .lock()
            .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?;
        let waited = state.observe_demand(id).is_pending();
        while state.is_pending(id) {
            state = self
                .shared
                .1
                .wait(state)
                .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?;
        }
        Ok(state.resolve_demand(id, waited.then(|| started.elapsed()))?)
    }

    /// Cancels queued work, fences in-flight work, and returns its first failure.
    pub fn cancel(&self) -> Result<(), BackgroundPrefetchWorkerError> {
        let mut state = self
            .shared
            .0
            .lock()
            .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?;
        state.cancel_all()?;
        self.shared.1.notify_all();
        while !state.is_idle() {
            state = self
                .shared
                .1
                .wait(state)
                .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?;
        }
        let failure = state.finish_cancellation()?;
        self.shared.1.notify_all();
        match failure {
            Some((id, message)) => {
                Err(BackgroundPrefetchWorkerError::OperationFailed { id, message })
            }
            None => Ok(()),
        }
    }

    /// Returns an immutable lifecycle and backpressure report.
    pub fn report(&self) -> Result<BackgroundPrefetchReport, BackgroundPrefetchWorkerError> {
        Ok(self
            .shared
            .0
            .lock()
            .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?
            .report())
    }

    /// Waits until all admitted work reaches a terminal state.
    pub fn wait_idle(&self) -> Result<(), BackgroundPrefetchWorkerError> {
        let mut state = self
            .shared
            .0
            .lock()
            .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?;
        while !state.is_idle() {
            state = self
                .shared
                .1
                .wait(state)
                .map_err(|_| BackgroundPrefetchWorkerError::StatePoisoned)?;
        }
        Ok(())
    }
}

impl Drop for BackgroundPrefetchWorker {
    fn drop(&mut self) {
        if self.nonblocking_drop {
            self.stopping.store(true, Ordering::Release);
        } else {
            let _ = self.cancel();
            let _ = self.sender.send(WorkerMessage::Shutdown);
        }
        if let Some(worker) = self.worker.take() {
            if !self.nonblocking_drop {
                let _ = worker.join();
            }
        }
    }
}

fn worker_loop(
    operation: PrefetchOperation,
    receiver: mpsc::Receiver<WorkerMessage>,
    shared: Arc<(Mutex<PrefetchExecutionState<String>>, Condvar)>,
    stopping: Arc<AtomicBool>,
) {
    while let Ok(message) = receiver.recv() {
        if stopping.load(Ordering::Acquire) {
            break;
        }
        let WorkerMessage::WorkAvailable = message else {
            break;
        };
        let work = {
            let Ok(mut state) = shared.0.lock() else {
                break;
            };
            let work = state.begin_next();
            shared.1.notify_all();
            work
        };
        let Some(work) = work else {
            continue;
        };
        let result = catch_unwind(AssertUnwindSafe(|| operation(work.id())))
            .map_err(|payload| {
                payload
                    .downcast_ref::<&str>()
                    .map(|message| (*message).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "background prefetch operation panicked".to_string())
            })
            .and_then(|result| result);
        let Ok(mut state) = shared.0.lock() else {
            break;
        };
        state
            .complete(work, result)
            .expect("worker completion matches runtime-owned admitted work");
        shared.1.notify_all();
    }
    if stopping.load(Ordering::Acquire) {
        if let Ok(mut state) = shared.0.lock() {
            let _ = state.cancel_all();
            let _ = state.finish_cancellation();
            shared.1.notify_all();
        }
    }
}

/// Failure from the backend-neutral background prefetch worker.
#[derive(Debug, thiserror::Error)]
pub enum BackgroundPrefetchWorkerError {
    /// Shared worker state was poisoned.
    #[error("background prefetch worker state is poisoned")]
    StatePoisoned,
    /// The worker ended before accepting or completing required work.
    #[error("background prefetch worker disconnected")]
    WorkerDisconnected,
    /// A backend operation failed and was retained for demand or cancellation.
    #[error("background prefetch of {id} failed: {message}")]
    OperationFailed {
        /// Failed residency unit.
        id: OffloadUnitId,
        /// Original backend failure.
        message: String,
    },
    /// Backend-neutral lifecycle misuse.
    #[error(transparent)]
    State(#[from] eredu_core::residency::PrefetchStateError),
    /// Worker creation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn id(value: &str) -> OffloadUnitId {
        OffloadUnitId::new(value).unwrap()
    }

    #[test]
    fn worker_coalesces_contains_panics_and_reports_demand() {
        let worker = BackgroundPrefetchWorker::new(2, "runtime-prefetch-test", |id| {
            if id.as_str() == "panic" {
                panic!("controlled prefetch panic");
            }
            Ok(())
        })
        .unwrap();
        let ready = id("ready");
        worker.submit(&ready, false).unwrap();
        worker.submit(&ready, false).unwrap();
        assert_eq!(
            worker.wait(&ready).unwrap(),
            PrefetchDemandResolution::Ready
        );

        let panic = id("panic");
        worker.submit(&panic, false).unwrap();
        let resolution = worker.wait(&panic).unwrap();
        assert!(
            matches!(resolution, PrefetchDemandResolution::Failed(message) if message.contains("controlled prefetch panic"))
        );
        let report = worker.report().unwrap();
        assert_eq!(report.submitted(), 2);
        assert!(report.coalesced() >= 1);
        assert_eq!(report.completed(), 1);
        assert_eq!(report.failed(), 1);
    }

    #[test]
    fn drop_cancels_and_joins_in_flight_work() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let operation_gate = Arc::clone(&gate);
        let worker = BackgroundPrefetchWorker::new(1, "runtime-prefetch-drop", move |_| {
            let mut released = operation_gate.0.lock().unwrap();
            while !*released {
                released = operation_gate.1.wait(released).unwrap();
            }
            Ok(())
        })
        .unwrap();
        worker.submit(&id("layer"), false).unwrap();
        while worker.report().unwrap().started() == 0 {
            thread::yield_now();
        }
        let (finished_tx, finished_rx) = mpsc::channel();
        thread::spawn(move || {
            drop(worker);
            finished_tx.send(()).unwrap();
        });
        assert!(finished_rx.recv_timeout(Duration::from_millis(20)).is_err());
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        finished_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    }

    #[test]
    fn nonblocking_drop_retains_active_operation_and_cancels_queued_work() {
        struct OperationOwner(mpsc::Sender<()>);
        impl Drop for OperationOwner {
            fn drop(&mut self) {
                let _ = self.0.send(());
            }
        }

        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let operation_gate = Arc::clone(&gate);
        let (started_tx, started_rx) = mpsc::channel();
        let (released_tx, released_rx) = mpsc::channel();
        let owner = OperationOwner(released_tx);
        let worker = BackgroundPrefetchWorker::new(1, "runtime-prefetch-detach", move |_| {
            let _retained = &owner;
            let _ = started_tx.send(());
            let mut released = operation_gate.0.lock().unwrap();
            while !*released {
                released = operation_gate.1.wait(released).unwrap();
            }
            Ok(())
        })
        .unwrap()
        .with_nonblocking_drop();
        worker.submit(&id("active"), false).unwrap();
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        worker.submit(&id("queued"), false).unwrap();
        let shared = Arc::clone(&worker.shared);
        let (dropped_tx, dropped_rx) = mpsc::channel();
        thread::spawn(move || {
            drop(worker);
            let _ = dropped_tx.send(());
        });
        let dropped_before_release = dropped_rx.recv_timeout(Duration::from_secs(1));
        let owner_still_retained = released_rx.try_recv().is_err();
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        dropped_before_release.unwrap();
        assert!(owner_still_retained);
        released_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let report = shared.0.lock().unwrap().report();
        assert_eq!(report.started(), 1);
        assert_eq!(report.cancelled(), 1);
        assert!(started_rx.try_recv().is_err());
    }

    #[test]
    fn cancellation_fences_active_and_queued_generations() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let operation_gate = Arc::clone(&gate);
        let worker = Arc::new(
            BackgroundPrefetchWorker::new(1, "runtime-prefetch-cancel", move |_| {
                let mut released = operation_gate.0.lock().unwrap();
                while !*released {
                    released = operation_gate.1.wait(released).unwrap();
                }
                Ok(())
            })
            .unwrap(),
        );
        worker.submit(&id("active"), false).unwrap();
        while worker.report().unwrap().started() == 0 {
            thread::yield_now();
        }
        worker.submit(&id("queued"), false).unwrap();

        let cancelling = Arc::clone(&worker);
        let (cancelled_tx, cancelled_rx) = mpsc::channel();
        thread::spawn(move || cancelled_tx.send(cancelling.cancel()).unwrap());
        let mut state = worker.shared.0.lock().unwrap();
        while state.generation() == 0 {
            state = worker.shared.1.wait(state).unwrap();
        }
        drop(state);
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        cancelled_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        let report = worker.report().unwrap();
        assert_eq!(report.started(), 1);
        assert_eq!(report.completed(), 0);
        assert_eq!(report.cancelled(), 2);
    }

    #[test]
    fn disconnected_notification_rolls_back_admission() {
        let mut worker =
            BackgroundPrefetchWorker::new(1, "runtime-prefetch-disconnect", |_| Ok(())).unwrap();
        worker.sender.send(WorkerMessage::Shutdown).unwrap();
        worker.worker.take().unwrap().join().unwrap();
        assert!(matches!(
            worker.submit(&id("layer"), false),
            Err(BackgroundPrefetchWorkerError::WorkerDisconnected)
        ));
        assert_eq!(worker.report().unwrap().submitted(), 0);
        worker.cancel().unwrap();
    }
}

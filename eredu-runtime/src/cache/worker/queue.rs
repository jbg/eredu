//! One explicit finite physical FIFO for ordinary and paid worker sources.
use super::*;
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError, WorkspaceMetadataFunding},
};
use std::{
    collections::VecDeque,
    mem::{size_of, size_of_val},
    sync::atomic::AtomicUsize,
};

type Request<Task, Output> = CacheIoWorkerRequest<Task, Output>;
struct Storage<Task, Output> {
    values: VecDeque<Request<Task, Output>>,
    maximum: usize,
    prepared: bool,
    // Every queued task retires before this destination's source custody.
    funding: Option<WorkspaceMetadataFunding>,
}
struct State<Task, Output> {
    storage: Storage<Task, Output>,
    receiver: bool,
    closed: bool,
}
struct Shared<Task, Output> {
    state: Mutex<State<Task, Output>>,
    ready: Condvar,
    senders: AtomicUsize,
}
pub(super) struct Sender<Task, Output> {
    shared: Arc<Shared<Task, Output>>,
}
pub(super) struct Receiver<Task, Output> {
    shared: Arc<Shared<Task, Output>>,
    worker: Arc<CacheIoWorkerShared<Output>>,
}
pub(super) fn channel<Task, Output>(
    maximum: usize,
    worker: Arc<CacheIoWorkerShared<Output>>,
) -> (Sender<Task, Output>, Receiver<Task, Output>) {
    let shared = Arc::new(Shared {
        state: Mutex::new(State {
            storage: Storage {
                values: VecDeque::with_capacity(maximum),
                maximum,
                funding: None,
                prepared: false,
            },
            receiver: true,
            closed: false,
        }),
        ready: Condvar::new(),
        senders: AtomicUsize::new(1),
    });
    (
        Sender {
            shared: Arc::clone(&shared),
        },
        Receiver { shared, worker },
    )
}
impl<Task, Output> Clone for Sender<Task, Output> {
    fn clone(&self) -> Self {
        let count = self.shared.senders.fetch_add(1, Ordering::Relaxed);
        assert!(
            count < isize::MAX as usize,
            "cache I/O sender count overflow"
        );
        Self {
            shared: Arc::clone(&self.shared),
        }
    }
}
impl<Task, Output> Drop for Sender<Task, Output> {
    fn drop(&mut self) {
        if self.shared.senders.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        // Queue critical sections never call a backend or drop task payloads.
        // Predicate mutation and notification share the receiver's mutex.
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|cause| cause.into_inner());
        state.closed = true;
        self.shared.ready.notify_all();
    }
}
pub(super) struct SendFailure<Task, Output> {
    pub(super) cause: CacheIoExecutionStateError,
    pub(super) request: Request<Task, Output>,
}
impl<Task, Output> Sender<Task, Output> {
    pub(super) fn send(
        &self,
        request: Request<Task, Output>,
    ) -> Result<(), SendFailure<Task, Output>> {
        let mut state = match self.shared.state.lock() {
            Ok(state) => state,
            Err(_) => {
                return Err(SendFailure {
                    cause: CacheIoExecutionStateError::CoordinationPoisoned,
                    request,
                });
            }
        };
        let failure = if state.closed || !state.receiver {
            Some(CacheIoExecutionStateError::WorkerStopped)
        } else if state.storage.values.len() == state.storage.maximum {
            Some(CacheIoExecutionStateError::QueueCapacity)
        } else {
            None
        };
        if let Some(cause) = failure {
            drop(state);
            return Err(SendFailure { cause, request });
        }
        state.storage.values.push_back(request);
        self.shared.ready.notify_one();
        Ok(())
    }
}
impl<Task, Output> Receiver<Task, Output> {
    pub(super) fn recv(&self) -> Result<Request<Task, Output>, ()> {
        let mut state = self.shared.state.lock().map_err(|_| ())?;
        loop {
            if let Some(request) = state.storage.values.pop_front() {
                return Ok(request);
            }
            if state.closed {
                return Err(());
            }
            state = self.shared.ready.wait(state).map_err(|_| ())?;
        }
    }
}
impl<Task, Output> Drop for Receiver<Task, Output> {
    fn drop(&mut self) {
        self.worker.stopping.store(true, Ordering::Release);
        self.worker.space_available.notify_all();
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|cause| cause.into_inner());
        state.receiver = false;
        state.closed = true;
        let pending = std::mem::take(&mut state.storage.values);
        self.shared.ready.notify_all();
        drop(state);
        for request in pending {
            if let Request::Operation {
                key,
                task,
                completion,
            } = request
            {
                retire_stopped_task(key, task, completion, &self.worker);
            }
        }
    }
}

/// Paid physical queue backing for the same worker receive loop. Operation
/// admission, task/completion storage and actual I/O remain independently checked.
pub struct PreparedCacheIoQueue<Task, Output> {
    storage: Option<Storage<Task, Output>>,
    source: Arc<Shared<Task, Output>>,
    worker: Arc<CacheIoWorkerShared<Output>>,
}
/// Empty old queue storage returned after an atomic idle installation.
pub struct RetiredCacheIoQueue<Task, Output> {
    _storage: Storage<Task, Output>,
}
/// Atomic queue installation refusal, retaining the unchanged paid destination.
pub struct CacheIoQueueInstallationError<Task, Output> {
    cause: CacheIoRegistryRefusal,
    retained: PreparedCacheIoQueue<Task, Output>,
}
impl<Task, Output> CacheIoQueueInstallationError<Task, Output> {
    /// Exact source/idle refusal.
    pub fn cause(&self) -> CacheIoRegistryRefusal {
        self.cause
    }
    /// Recover the same destination for a later idle attempt.
    pub fn into_parts(self) -> (CacheIoRegistryRefusal, PreparedCacheIoQueue<Task, Output>) {
        (self.cause, self.retained)
    }
}
impl<Task, Output> std::fmt::Debug for PreparedCacheIoQueue<Task, Output> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedCacheIoQueue")
            .finish_non_exhaustive()
    }
}
impl<Task, Output> std::fmt::Debug for RetiredCacheIoQueue<Task, Output> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetiredCacheIoQueue")
            .finish_non_exhaustive()
    }
}
impl<Task, Output> std::fmt::Debug for CacheIoQueueInstallationError<Task, Output> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.cause, f)
    }
}
impl<Task, Output> std::fmt::Display for CacheIoQueueInstallationError<Task, Output> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl<Task: 'static, Output: 'static> std::error::Error
    for CacheIoQueueInstallationError<Task, Output>
{
}
impl<Task, Output> PreparedCacheIoQueue<Task, Output> {
    /// Exact backing plus fixed construction/installation controls for an actual
    /// queue capacity. One additional slot is reserved for orderly shutdown.
    pub fn control_bytes(queue_capacity: usize) -> Option<usize> {
        Self::fixed_bytes()?.checked_add(WorkspaceContext::metadata_vec_bytes::<
            Request<Task, Output>,
        >(queue_capacity.checked_add(1)?)?)
    }
    fn fixed_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Storage<Task, Output>>(),
            size_of::<RetiredCacheIoQueue<Task, Output>>(),
            size_of::<CacheIoQueueInstallationError<Task, Output>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<
                Result<
                    RetiredCacheIoQueue<Task, Output>,
                    CacheIoQueueInstallationError<Task, Output>,
                >,
            >(),
            size_of::<Result<(), std::collections::TryReserveError>>(),
            size_of::<VecDeque<Request<Task, Output>>>(),
            size_of::<(&WorkspaceContext, usize, usize)>(),
            size_of::<std::sync::MutexGuard<'_, State<Task, Output>>>(),
            size_of::<std::sync::MutexGuard<'_, CacheIoExecutionState>>(),
            size_of::<
                std::sync::MutexGuard<
                    '_,
                    CacheRecordTable<CacheIoOperationKey, CompletionOwner<Output>>,
                >,
            >(),
            WorkspaceContext::metadata_source_bytes::<CacheIoRegistryRefusal>()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Replace only the exact idle worker's queue. Source or idle refusal keeps
    /// both the live queue and prepared destination unchanged.
    pub fn install(
        mut self,
        worker: &CacheIoWorker<Task, Output>,
    ) -> Result<RetiredCacheIoQueue<Task, Output>, CacheIoQueueInstallationError<Task, Output>>
    {
        let result = (|| {
            if !Arc::ptr_eq(&self.source, &worker.sender.shared)
                || !Arc::ptr_eq(&self.worker, &worker.shared)
            {
                return Err(CacheIoRegistryRefusal::ForeignWorker);
            }
            if worker.shared.stopping.load(Ordering::Acquire) {
                return Err(CacheIoRegistryRefusal::Stopped);
            }
            let execution = worker.shared.execution.try_lock().map_err(lock_refusal)?;
            let completions = worker.shared.in_flight.try_lock().map_err(lock_refusal)?;
            let mut queue = worker
                .sender
                .shared
                .state
                .try_lock()
                .map_err(lock_refusal)?;
            if !execution.is_empty()
                || !completions.is_empty()
                || worker.shared.active_payload.load(Ordering::Acquire)
                || !queue.storage.values.is_empty()
            {
                return Err(CacheIoRegistryRefusal::Busy);
            }
            if !queue.receiver || queue.closed {
                return Err(CacheIoRegistryRefusal::Stopped);
            }
            Ok(RetiredCacheIoQueue {
                _storage: std::mem::replace(
                    &mut queue.storage,
                    self.storage.take().expect("one queue installation"),
                ),
            })
        })();
        result.map_err(|cause| CacheIoQueueInstallationError {
            cause,
            retained: self,
        })
    }
}
fn lock_refusal<T>(cause: std::sync::TryLockError<T>) -> CacheIoRegistryRefusal {
    match cause {
        std::sync::TryLockError::WouldBlock => CacheIoRegistryRefusal::Busy,
        std::sync::TryLockError::Poisoned(_) => CacheIoRegistryRefusal::Poisoned,
    }
}
impl<Task, Output> CacheIoWorker<Task, Output> {
    /// Prepare queue backing from this actual worker's unchanged capacity. The
    /// destination is not published until exact idle installation succeeds.
    pub fn prepare_queue(
        &self,
        context: &WorkspaceContext,
    ) -> Result<PreparedCacheIoQueue<Task, Output>, Error> {
        context.charge_metadata(
            PreparedCacheIoQueue::<Task, Output>::fixed_bytes()
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let capacity = self
            .shared
            .execution
            .try_lock()
            .map(|v| v.capacity())
            .map_err(|cause| context.metadata_source(lock_refusal(cause)))?;
        let maximum = capacity
            .checked_add(1)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        context.charge_metadata(
            WorkspaceContext::metadata_vec_bytes::<Request<Task, Output>>(maximum)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let mut values = VecDeque::new();
        values
            .try_reserve_exact(maximum)
            .map_err(|cause| context.metadata_source(cause))?;
        Ok(PreparedCacheIoQueue {
            storage: Some(Storage {
                values,
                maximum,
                funding: context.metadata_funding(),
                prepared: true,
            }),
            source: Arc::clone(&self.sender.shared),
            worker: Arc::clone(&self.shared),
        })
    }
}

#[cfg(test)]
#[path = "queue/tests.rs"]
mod tests;

impl<Task, Output> Sender<Task, Output> {
    pub(super) fn is_prepared(&self) -> Result<bool, CacheIoWorkerError> {
        let state = self
            .shared
            .state
            .lock()
            .map_err(|_| CacheIoWorkerError::Poisoned)?;
        Ok(state.storage.prepared)
    }
}

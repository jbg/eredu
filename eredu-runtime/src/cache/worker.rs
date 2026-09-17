//! Backend-neutral bounded physical worker for cache backing-store tasks.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
};

use super::{
    CacheIoAdmission, CacheIoCompletionDisposition, CacheIoExecutionState,
    CacheIoExecutionStateError, CacheIoOperationKey, CacheIoPreparation, CacheIoStartDisposition,
    CacheRecordTable,
};

#[path = "worker/registry.rs"]
mod registry;
pub use registry::{
    CacheIoRegistryInstallationError, CacheIoRegistryRefusal, PreparedCacheIoRegistry,
    RetiredCacheIoRegistry,
};

#[path = "worker/queue.rs"]
mod queue;
pub use queue::{CacheIoQueueInstallationError, PreparedCacheIoQueue, RetiredCacheIoQueue};

#[path = "worker/task.rs"]
mod task;
pub use task::{
    CacheIoBorrowedError, CacheIoTaskPreparationError, CacheIoTaskRefusal, PreparedCacheIoTask,
    PreparedCacheIoTaskSlot,
};
use task::{CompletionOwner, TaskInput};

enum CacheIoWorkerRequest<Task, Output> {
    Operation {
        key: CacheIoOperationKey,
        task: Box<Option<Task>>,
        completion: CompletionOwner<Output>,
    },
    Stop,
}

#[derive(Debug, Clone)]
enum CacheIoCompletionState<Output> {
    Finished(Result<Output, String>),
    WorkerFailure(CacheIoExecutionStateError),
    Cancelled,
}

#[derive(Debug)]
struct CacheIoCompletion<Output> {
    state: Mutex<Option<CacheIoCompletionState<Output>>>,
    ready: Condvar,
    released: Mutex<bool>,
    released_ready: Condvar,
}

impl<Output> Default for CacheIoCompletion<Output> {
    fn default() -> Self {
        Self {
            state: Mutex::new(None),
            ready: Condvar::new(),
            released: Mutex::new(false),
            released_ready: Condvar::new(),
        }
    }
}

impl<Output> CacheIoCompletion<Output> {
    fn finish(&self, result: Result<Output, String>) {
        if let Ok(mut state) = self.state.lock() {
            if state.is_none() {
                *state = Some(CacheIoCompletionState::Finished(result));
                self.ready.notify_all();
            }
        }
    }

    fn finish_failure(&self, cause: CacheIoExecutionStateError) {
        if let Ok(mut state) = self.state.lock() {
            if state.is_none() {
                *state = Some(CacheIoCompletionState::WorkerFailure(cause));
                self.ready.notify_all();
            }
        }
    }

    fn cancel(&self) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        if state.is_some() {
            return false;
        }
        *state = Some(CacheIoCompletionState::Cancelled);
        self.ready.notify_all();
        true
    }

    fn is_ready(&self) -> bool {
        self.state.lock().map_or(true, |state| state.is_some())
    }

    fn release_task_resources(&self) {
        if let Ok(mut released) = self.released.lock() {
            *released = true;
            self.released_ready.notify_all();
        }
    }

    fn wait_for_task_resources(&self) -> Result<(), CacheIoWorkerError> {
        let mut released = self
            .released
            .lock()
            .map_err(|_| CacheIoWorkerError::Poisoned)?;
        while !*released {
            released = self
                .released_ready
                .wait(released)
                .map_err(|_| CacheIoWorkerError::Poisoned)?;
        }
        Ok(())
    }
}

impl<Output: Clone> CacheIoCompletion<Output> {
    fn wait(&self, generation: u64) -> Result<Output, CacheIoWorkerError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| CacheIoWorkerError::Poisoned)?;
        while state.is_none() {
            state = self
                .ready
                .wait(state)
                .map_err(|_| CacheIoWorkerError::Poisoned)?;
        }
        match state.as_ref().expect("completion state was awaited") {
            CacheIoCompletionState::Finished(Ok(output)) => Ok(output.clone()),
            CacheIoCompletionState::Finished(Err(error)) => {
                Err(CacheIoWorkerError::OperationFailed(error.clone()))
            }
            CacheIoCompletionState::WorkerFailure(cause) => {
                Err(CacheIoWorkerError::Execution(*cause))
            }
            CacheIoCompletionState::Cancelled => Err(CacheIoWorkerError::Cancelled { generation }),
        }
    }
}

#[derive(Debug)]
struct CacheIoWorkerShared<Output> {
    in_flight: Mutex<CacheRecordTable<CacheIoOperationKey, CompletionOwner<Output>>>,
    execution: Mutex<CacheIoExecutionState>,
    space_available: Condvar,
    stopping: AtomicBool,
    shutdown_polling: AtomicBool,
    active_payload: AtomicBool,
}

impl<Output> CacheIoWorkerShared<Output> {
    fn new(capacity: usize) -> Result<Self, CacheIoWorkerError> {
        Ok(Self {
            in_flight: Mutex::new(CacheRecordTable::new()),
            execution: Mutex::new(CacheIoExecutionState::new(capacity)?),
            space_available: Condvar::new(),
            stopping: AtomicBool::new(false),
            shutdown_polling: AtomicBool::new(false),
            active_payload: AtomicBool::new(false),
        })
    }
}

struct ActivePayload<'a>(&'a AtomicBool);

impl Drop for ActivePayload<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// Exact completion ownership for one coalesced cache I/O operation.
pub struct CacheIoTicket<Output> {
    /// Exact logical operation identity.
    pub key: CacheIoOperationKey,
    completion: CompletionOwner<Output>,
    shared: Arc<CacheIoWorkerShared<Output>>,
}

impl<Output> Clone for CacheIoTicket<Output> {
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            completion: self.completion.clone(),
            shared: Arc::clone(&self.shared),
        }
    }
}

impl<Output> std::fmt::Debug for CacheIoTicket<Output> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CacheIoTicket")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

impl<Output: Clone> CacheIoTicket<Output> {
    /// Waits for the logical output or cancellation.
    pub fn wait(&self) -> Result<Output, CacheIoWorkerError> {
        self.completion.wait(self.key.generation)
    }

    /// Cancels prepared, queued, or in-flight work exactly once.
    pub fn cancel(&self) -> bool {
        let Ok(mut execution) = self.shared.execution.lock() else {
            return false;
        };
        let cancelled = execution.cancel(&self.key) && self.completion.cancel();
        self.shared.space_available.notify_all();
        cancelled
    }

    /// Waits until all backend task inputs and retained resources are dropped.
    pub fn wait_for_task_resources(&self) -> Result<(), CacheIoWorkerError> {
        self.completion.wait_for_task_resources()
    }

    /// Returns whether two tickets join the same exact completion owner.
    pub fn shares_completion_with(&self, other: &Self) -> bool {
        self.completion.same(&other.completion)
    }
}

/// Prepared cache I/O that admits physical work only when explicitly enqueued.
pub struct CacheIoSubmission<Task, Output> {
    /// Ticket shared by the operation owner and all exact-key joiners.
    pub ticket: CacheIoTicket<Output>,
    sender: queue::Sender<Task, Output>,
    shared: Arc<CacheIoWorkerShared<Output>>,
    unsent: Option<CacheIoWorkerRequest<Task, Output>>,
    joined_task: Option<TaskInput<Task, Output>>,
    /// Whether this submission joined an already prepared exact operation.
    pub joined: bool,
}

/// Physical admission observations for one submission.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct CacheIoSubmissionOutcome {
    /// Whether this submission joined an existing exact operation.
    pub joined: bool,
    /// Whether finite queue capacity delayed physical admission.
    pub backpressure: bool,
    /// Largest observed physical queue occupancy.
    pub peak_occupancy: usize,
}

impl<Task, Output: Clone> CacheIoSubmission<Task, Output> {
    /// Returns the unused backend task when this submission joined existing work.
    ///
    /// Backends may disarm task-local rollback guards before the unused task is
    /// dropped; the task is never physically executed.
    pub fn joined_task_mut(&mut self) -> Option<&mut Task> {
        self.joined_task.as_mut().map(TaskInput::task_mut)
    }

    /// Admits this prepared task, blocking only on finite queue capacity.
    pub fn enqueue(mut self) -> Result<CacheIoSubmissionOutcome, CacheIoWorkerError> {
        let mut backpressure = false;
        if let Some(request) = self.unsent.take() {
            let mut execution = match self.shared.execution.lock() {
                Ok(execution) => execution,
                Err(_) => {
                    drop(request);
                    self.ticket.completion.release_task_resources();
                    return Err(CacheIoWorkerError::Poisoned);
                }
            };
            loop {
                if self.shared.stopping.load(Ordering::Acquire) {
                    execution.cancel(&self.ticket.key);
                    drop(execution);
                    drop(request);
                    self.ticket
                        .completion
                        .finish_failure(CacheIoExecutionStateError::WorkerStopped);
                    self.ticket.completion.release_task_resources();
                    retire_completion(&self.shared, &self.ticket.key, &self.ticket.completion);
                    return Err(CacheIoExecutionStateError::WorkerStopped.into());
                }
                match execution.admit(&self.ticket.key)? {
                    CacheIoAdmission::Admitted => {
                        if let Err(failure) = self.sender.send(request) {
                            let rollback = execution.rollback_admission(&self.ticket.key);
                            drop(execution);
                            drop(failure.request);
                            self.ticket.completion.finish_failure(failure.cause);
                            self.ticket.completion.release_task_resources();
                            retire_completion(
                                &self.shared,
                                &self.ticket.key,
                                &self.ticket.completion,
                            );
                            rollback?;
                            return Err(failure.cause.into());
                        }
                        break;
                    }
                    CacheIoAdmission::AtCapacity => {
                        backpressure = true;
                        // Opt-in shutdown cannot lock execution from Drop:
                        // backend task cleanup may itself need a native lock.
                        // A bounded check also closes a shutdown notification
                        // racing between the predicate check and this wait.
                        let waited = if self.shared.shutdown_polling.load(Ordering::Acquire) {
                            self.shared
                                .space_available
                                .wait_timeout(execution, std::time::Duration::from_millis(25))
                                .map(|(execution, _)| execution)
                                .map_err(|_| ())
                        } else {
                            self.shared.space_available.wait(execution).map_err(|_| ())
                        };
                        execution = match waited {
                            Ok(execution) => execution,
                            Err(_) => {
                                drop(request);
                                self.ticket.completion.release_task_resources();
                                return Err(CacheIoWorkerError::Poisoned);
                            }
                        };
                    }
                    CacheIoAdmission::Cancelled => {
                        let peak_occupancy = execution.peak_queued();
                        drop(execution);
                        drop(request);
                        self.ticket.completion.release_task_resources();
                        return Ok(CacheIoSubmissionOutcome {
                            joined: self.joined,
                            backpressure,
                            peak_occupancy,
                        });
                    }
                }
            }
            drop(execution);
        }
        Ok(CacheIoSubmissionOutcome {
            joined: self.joined,
            backpressure,
            peak_occupancy: self
                .shared
                .execution
                .lock()
                .map_err(|_| CacheIoWorkerError::Poisoned)?
                .peak_queued(),
        })
    }
}

impl<Task, Output> Drop for CacheIoSubmission<Task, Output> {
    fn drop(&mut self) {
        let Some(request) = self.unsent.take() else {
            return;
        };
        if let Ok(mut execution) = self.shared.execution.lock() {
            execution.cancel(&self.ticket.key);
        }
        drop(request);
        self.ticket.completion.release_task_resources();
        retire_completion(&self.shared, &self.ticket.key, &self.ticket.completion);
    }
}

/// Generic bounded background worker over opaque backend task and output types.
pub struct CacheIoWorker<Task, Output> {
    sender: queue::Sender<Task, Output>,
    handle: Mutex<Option<JoinHandle<()>>>,
    shared: Arc<CacheIoWorkerShared<Output>>,
    nonblocking_drop: bool,
}

impl<Task, Output> std::fmt::Debug for CacheIoWorker<Task, Output> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CacheIoWorker")
            .finish_non_exhaustive()
    }
}

impl<Task, Output> CacheIoWorker<Task, Output>
where
    Task: Send + 'static,
    Output: Clone + Send + 'static,
{
    /// Starts a bounded worker using statically dispatched task and cleanup functions.
    pub fn new(
        capacity: usize,
        thread_name: impl Into<String>,
        execute: fn(Task) -> Result<Output, String>,
        discard: fn(Output),
    ) -> Result<Self, CacheIoWorkerError> {
        let thread_name = thread_name.into();
        let maximum = capacity
            .checked_add(1)
            .ok_or(CacheIoExecutionStateError::QueueSizeOverflow)?;
        let shared = Arc::new(CacheIoWorkerShared::new(capacity)?);
        let (sender, receiver) = queue::channel(maximum, Arc::clone(&shared));
        let worker_shared = Arc::clone(&shared);
        let handle = thread::Builder::new()
            .name(thread_name.clone())
            .spawn(move || {
                while let Ok(request) = receiver.recv() {
                    worker_shared.active_payload.store(true, Ordering::Release);
                    // This outer guard drops after the match's task, output and
                    // completion owners, including all early-continue paths.
                    let _active = ActivePayload(&worker_shared.active_payload);
                    match request {
                        CacheIoWorkerRequest::Operation {
                            key,
                            task,
                            completion,
                        } => {
                            if worker_shared.stopping.load(Ordering::Acquire) {
                                retire_stopped_task(key, task, completion, &worker_shared);
                                continue;
                            }
                            let start = worker_shared
                                .execution
                                .lock()
                                .map_err(|_| CacheIoExecutionStateError::CoordinationPoisoned)
                                .and_then(|mut execution| execution.begin(&key));
                            worker_shared.space_available.notify_all();
                            match start {
                                Ok(CacheIoStartDisposition::Execute) => {}
                                Ok(CacheIoStartDisposition::Discard) => {
                                    drop(task);
                                    completion.release_task_resources();
                                    retire_completion(&worker_shared, &key, &completion);
                                    continue;
                                }
                                Err(error) => {
                                    drop(task);
                                    completion.finish_failure(error);
                                    completion.release_task_resources();
                                    retire_completion(&worker_shared, &key, &completion);
                                    continue;
                                }
                            }
                            let result = catch_unwind(AssertUnwindSafe(|| {
                                execute((*task).expect("bound worker task"))
                            }));
                            let disposition = worker_shared
                                .execution
                                .lock()
                                .map_err(|_| CacheIoWorkerError::Poisoned)
                                .and_then(|mut execution| {
                                    execution.complete(&key).map_err(Into::into)
                                });
                            if !matches!(disposition, Ok(CacheIoCompletionDisposition::Publish))
                                || completion.is_ready()
                            {
                                if let Ok(Ok(output)) = result {
                                    discard(output);
                                }
                            } else {
                                match result {
                                    Ok(result) => completion.finish(result),
                                    Err(_) => completion
                                        .finish_failure(CacheIoExecutionStateError::TaskPanicked),
                                }
                            }
                            // A successful wait permits a new occurrence of this
                            // exact key. Retire its registry entry before notifying.
                            retire_completion(&worker_shared, &key, &completion);
                            completion.release_task_resources();
                        }
                        CacheIoWorkerRequest::Stop => break,
                    }
                }
            })
            .map_err(|source| CacheIoWorkerError::Spawn {
                thread_name,
                source,
            })?;
        Ok(Self {
            sender,
            handle: Mutex::new(Some(handle)),
            shared,
            nonblocking_drop: false,
        })
    }

    /// Requests shutdown without joining from this handle's destructor.
    ///
    /// The worker owns in-flight tasks until they finish and discards queued
    /// tasks itself. Prepared submissions reject admission after shutdown.
    pub fn with_nonblocking_drop(mut self) -> Self {
        self.nonblocking_drop = true;
        self.shared.shutdown_polling.store(true, Ordering::Release);
        self
    }

    /// Prepares new work or joins an exact operation already owned by the worker.
    pub fn prepare(
        &self,
        key: CacheIoOperationKey,
        task: Task,
    ) -> Result<CacheIoSubmission<Task, Output>, CacheIoWorkerError> {
        let mut input = Some(TaskInput::Ordinary(task));
        self.prepare_input(key, &mut input, false)
    }

    fn prepare_input(
        &self,
        key: CacheIoOperationKey,
        input: &mut Option<TaskInput<Task, Output>>,
        require_prepared: bool,
    ) -> Result<CacheIoSubmission<Task, Output>, CacheIoWorkerError> {
        // Input ownership stays with the caller until all registry checks pass.
        // Every error therefore releases locks before backend payload teardown.
        let mut execution = self
            .shared
            .execution
            .lock()
            .map_err(|_| CacheIoWorkerError::Poisoned)?;
        let mut completions = self
            .shared
            .in_flight
            .lock()
            .map_err(|_| CacheIoWorkerError::Poisoned)?;
        let finite = execution.is_prepared();
        if require_prepared {
            if self.shared.stopping.load(Ordering::Acquire) {
                return Err(CacheIoExecutionStateError::WorkerStopped.into());
            }
            if !finite || !self.sender.is_prepared()? {
                return Err(CacheIoExecutionStateError::RegistryCapacity(
                    super::CacheTableCapacityError::Unprepared,
                )
                .into());
            }
        }
        let preparation = execution.try_prepare(key.clone())?;
        if preparation == CacheIoPreparation::Joined {
            let completion = completions
                .get(&key)
                .expect("runtime joined key has an exact completion");
            if require_prepared && !completion.is_prepared() {
                return Err(CacheIoExecutionStateError::RegistryCapacity(
                    super::CacheTableCapacityError::Unprepared,
                )
                .into());
            }
            return Ok(CacheIoSubmission {
                ticket: CacheIoTicket {
                    key,
                    completion: completion.clone(),
                    shared: Arc::clone(&self.shared),
                },
                sender: self.sender.clone(),
                shared: Arc::clone(&self.shared),
                unsent: None,
                joined_task: input.take(),
                joined: true,
            });
        }
        let completion = input.as_ref().expect("one task preparation").completion();
        if finite {
            if let Err((cause, _, retained)) =
                completions.insert_prepared(key.clone(), completion.clone())
            {
                execution.retire(&key)?;
                drop(completions);
                drop(execution);
                drop(retained);
                return Err(CacheIoExecutionStateError::RegistryCapacity(cause).into());
            }
        } else {
            completions.insert(key.clone(), completion.clone());
        }
        drop(completions);
        drop(execution);
        let task = input.take().expect("one task preparation").into_task();
        let request = CacheIoWorkerRequest::Operation {
            key: key.clone(),
            task,
            completion: completion.clone(),
        };
        Ok(CacheIoSubmission {
            ticket: CacheIoTicket {
                key,
                completion,
                shared: Arc::clone(&self.shared),
            },
            sender: self.sender.clone(),
            shared: Arc::clone(&self.shared),
            unsent: Some(request),
            joined_task: None,
            joined: false,
        })
    }

    /// Cold evidence that a prepared, queued or active task can retain payloads.
    /// Stays true through completion publication until worker-owned roots drop.
    /// This does not poll, retire or otherwise advance work.
    pub fn has_retained_work(&self) -> Result<bool, CacheIoWorkerError> {
        let pending = !self
            .shared
            .in_flight
            .lock()
            .map_err(|_| CacheIoWorkerError::Poisoned)?
            .is_empty();
        Ok(pending || self.shared.active_payload.load(Ordering::Acquire))
    }

    /// Nonblocking form of [`Self::has_retained_work`]. `None` means the
    /// retained-work registry is busy, never that the worker owns no payload.
    /// This only borrows the registry and reads the active-payload flag; it
    /// neither allocates a snapshot nor polls, cancels or retires a task.
    /// The result is point-in-time evidence, not a barrier against new work.
    pub fn try_has_retained_work(&self) -> Result<Option<bool>, CacheIoWorkerError> {
        let registry = match self.shared.in_flight.try_lock() {
            Ok(registry) => registry,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(None),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                return Err(CacheIoWorkerError::Poisoned);
            }
        };
        Ok(Some(
            !registry.is_empty() || self.shared.active_payload.load(Ordering::Acquire),
        ))
    }

    /// Releases exact-key ownership after task resources are safe to drop.
    pub fn retire(&self, ticket: &CacheIoTicket<Output>) {
        retire_completion(&self.shared, &ticket.key, &ticket.completion);
    }
}

impl<Task, Output> Drop for CacheIoWorker<Task, Output> {
    fn drop(&mut self) {
        if self.nonblocking_drop {
            self.shared.stopping.store(true, Ordering::Release);
            self.shared.space_available.notify_all();
            // Do not put Stop ahead of a racing prepared sender. Disconnect
            // after every remaining prepared handle has rejected admission;
            // the receiver owns and resolves any already-enqueued message.
        } else {
            let _ = self.sender.send(CacheIoWorkerRequest::Stop);
        }
        if let Ok(handle) = self.handle.get_mut() {
            if let Some(handle) = handle.take() {
                if !self.nonblocking_drop {
                    let _ = handle.join();
                }
            }
        }
    }
}

fn retire_stopped_task<Task, Output>(
    key: CacheIoOperationKey,
    task: Box<Option<Task>>,
    completion: CompletionOwner<Output>,
    shared: &CacheIoWorkerShared<Output>,
) {
    if let Ok(mut execution) = shared.execution.lock() {
        execution.cancel(&key);
        let _ = execution.begin(&key);
    }
    drop(task);
    completion.finish_failure(CacheIoExecutionStateError::WorkerStopped);
    retire_completion(shared, &key, &completion);
    completion.release_task_resources();
}

fn retire_completion<Output>(
    shared: &CacheIoWorkerShared<Output>,
    key: &CacheIoOperationKey,
    completion: &CompletionOwner<Output>,
) {
    let retired = if let Ok(mut execution) = shared.execution.lock() {
        execution.retire(key).unwrap_or(false)
    } else {
        false
    };
    if retired {
        shared.space_available.notify_all();
        if let Ok(mut in_flight) = shared.in_flight.lock() {
            if in_flight
                .get(key)
                .is_some_and(|current| current.same(completion))
            {
                in_flight.remove(key);
            }
        }
    }
}

/// Failure in generic cache I/O worker coordination or task execution.
#[derive(Debug, thiserror::Error)]
pub enum CacheIoWorkerError {
    /// The worker's synchronization state was poisoned.
    #[error("cache I/O worker synchronization state is poisoned")]
    Poisoned,
    /// A task returned a backend-specific failure string.
    #[error("cache I/O operation failed: {0}")]
    OperationFailed(String),
    /// Cancellation won for this generation.
    #[error("cache I/O operation was cancelled for generation {generation}")]
    Cancelled {
        /// Cancelled model/cache generation.
        generation: u64,
    },
    /// The physical worker thread could not be started.
    #[error("failed to start cache I/O worker {thread_name}: {source}")]
    Spawn {
        /// Requested worker thread name.
        thread_name: String,
        /// Underlying thread creation failure.
        #[source]
        source: std::io::Error,
    },
    /// The exact admission/cancellation state transition was invalid.
    #[error(transparent)]
    Execution(#[from] CacheIoExecutionStateError),
}

#[cfg(test)]
#[path = "worker/test_support.rs"]
pub(super) mod test_support;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CacheIoOperationKind;
    use eredu_core::cache::{CacheBlockId, CacheRepresentation};
    use std::{sync::mpsc, time::Duration};

    enum Task {
        Value(u64),
        Pause(mpsc::Sender<()>, mpsc::Receiver<()>),
        Panic,
    }

    fn execute(task: Task) -> Result<u64, String> {
        match task {
            Task::Value(value) => Ok(value),
            Task::Pause(started, release) => {
                let _ = started.send(());
                let _ = release.recv();
                Ok(0)
            }
            Task::Panic => panic!("injected worker panic"),
        }
    }

    fn discard(_value: u64) {}

    fn key(block: i64) -> CacheIoOperationKey {
        CacheIoOperationKey {
            generation: 7,
            id: CacheBlockId {
                session_id: 1,
                global_layer: 0,
                representation: CacheRepresentation::KeyValue,
                start: block,
                end: block + 1,
                rank: None,
            },
            kind: CacheIoOperationKind::Read,
        }
    }

    #[test]
    fn worker_coalesces_and_contains_task_panics() {
        let worker = CacheIoWorker::new(1, "cache-worker-test", execute, discard).unwrap();
        let first = worker.prepare(key(0), Task::Value(9)).unwrap();
        let first_ticket = first.ticket.clone();
        let joined = worker.prepare(key(0), Task::Value(10)).unwrap();
        let joined_ticket = joined.ticket.clone();
        assert!(joined.joined);
        first.enqueue().unwrap();
        joined.enqueue().unwrap();
        assert_eq!(first_ticket.wait().unwrap(), 9);
        assert_eq!(joined_ticket.wait().unwrap(), 9);
        assert!(first_ticket.shares_completion_with(&joined_ticket));
        worker.retire(&first_ticket);

        let panicking = worker.prepare(key(1), Task::Panic).unwrap();
        let ticket = panicking.ticket.clone();
        panicking.enqueue().unwrap();
        assert!(matches!(
            ticket.wait(),
            Err(CacheIoWorkerError::Execution(
                CacheIoExecutionStateError::TaskPanicked
            ))
        ));
        worker.retire(&ticket);
    }

    #[test]
    fn cancellation_wakes_a_backpressured_submission() {
        let worker =
            Arc::new(CacheIoWorker::new(1, "cache-worker-cancel-test", execute, discard).unwrap());
        assert!(!worker.has_retained_work().unwrap());
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let blocker = worker
            .prepare(key(0), Task::Pause(started_tx, release_rx))
            .unwrap();
        let blocker_ticket = blocker.ticket.clone();
        assert!(
            worker.has_retained_work().unwrap(),
            "prepared payload is retained before enqueue"
        );
        blocker.enqueue().unwrap();
        started_rx.recv().unwrap();
        assert!(
            worker.has_retained_work().unwrap(),
            "active worker retains the task"
        );

        let queued = worker.prepare(key(1), Task::Value(1)).unwrap();
        queued.enqueue().unwrap();
        let blocked = worker.prepare(key(2), Task::Value(2)).unwrap();
        let blocked_ticket = blocked.ticket.clone();
        let (outcome_tx, outcome_rx) = mpsc::channel();
        let enqueue = std::thread::spawn(move || outcome_tx.send(blocked.enqueue()).unwrap());
        assert!(outcome_rx.recv_timeout(Duration::from_millis(20)).is_err());
        assert!(blocked_ticket.cancel());
        assert!(
            outcome_rx
                .recv_timeout(Duration::from_secs(1))
                .unwrap()
                .unwrap()
                .backpressure
        );
        enqueue.join().unwrap();
        assert!(matches!(
            blocked_ticket.wait(),
            Err(CacheIoWorkerError::Cancelled { generation: 7 })
        ));
        release_tx.send(()).unwrap();
        assert_eq!(blocker_ticket.wait().unwrap(), 0);
    }

    #[test]
    fn nonblocking_drop_retains_active_task_and_retires_queued_work() {
        let worker = CacheIoWorker::new(1, "cache-worker-detach", execute, discard)
            .unwrap()
            .with_nonblocking_drop();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let active = worker
            .prepare(key(0), Task::Pause(started_tx, release_rx))
            .unwrap();
        let active_ticket = active.ticket.clone();
        active.enqueue().unwrap();
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let queued = worker.prepare(key(1), Task::Value(1)).unwrap();
        let queued_ticket = queued.ticket.clone();
        queued.enqueue().unwrap();
        let prepared = worker.prepare(key(2), Task::Value(2)).unwrap();
        let prepared_ticket = prepared.ticket.clone();

        let (dropped_tx, dropped_rx) = mpsc::channel();
        thread::spawn(move || {
            drop(worker);
            let _ = dropped_tx.send(());
        });
        let dropped_before_release = dropped_rx.recv_timeout(Duration::from_secs(1));
        let active_retained = !*active_ticket.completion.released.lock().unwrap();
        let rejected_prepared = prepared.enqueue();
        let prepared_released = *prepared_ticket.completion.released.lock().unwrap();
        release_tx.send(()).unwrap();
        dropped_before_release.unwrap();
        assert!(active_retained);
        assert!(matches!(
            rejected_prepared,
            Err(CacheIoWorkerError::Execution(
                CacheIoExecutionStateError::WorkerStopped
            ))
        ));
        assert!(prepared_released);
        assert!(matches!(
            prepared_ticket.wait(),
            Err(CacheIoWorkerError::Execution(
                CacheIoExecutionStateError::WorkerStopped
            ))
        ));
        assert_eq!(active_ticket.wait().unwrap(), 0);
        assert!(matches!(
            queued_ticket.wait(),
            Err(CacheIoWorkerError::Execution(
                CacheIoExecutionStateError::WorkerStopped
            ))
        ));
        active_ticket.wait_for_task_resources().unwrap();
        queued_ticket.wait_for_task_resources().unwrap();
        assert!(queued_ticket.shared.in_flight.lock().unwrap().is_empty());
    }

    #[test]
    fn nonblocking_shutdown_wakes_backpressured_submission_before_active_task_finishes() {
        let worker = CacheIoWorker::new(1, "cache-worker-detach-backpressure", execute, discard)
            .unwrap()
            .with_nonblocking_drop();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let active = worker
            .prepare(key(0), Task::Pause(started_tx, release_rx))
            .unwrap();
        active.enqueue().unwrap();
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        worker
            .prepare(key(1), Task::Value(1))
            .unwrap()
            .enqueue()
            .unwrap();
        let blocked = worker.prepare(key(2), Task::Value(2)).unwrap();
        let (outcome_tx, outcome_rx) = mpsc::channel();
        thread::spawn(move || {
            let _ = outcome_tx.send(blocked.enqueue());
        });
        assert!(outcome_rx.recv_timeout(Duration::from_millis(20)).is_err());
        drop(worker);
        let outcome_before_release = outcome_rx.recv_timeout(Duration::from_secs(1));
        release_tx.send(()).unwrap();
        assert!(matches!(
            outcome_before_release.unwrap(),
            Err(CacheIoWorkerError::Execution(
                CacheIoExecutionStateError::WorkerStopped
            ))
        ));
    }
}

#[cfg(test)]
#[path = "worker/retained_inspection_tests.rs"]
mod retained_inspection_tests;

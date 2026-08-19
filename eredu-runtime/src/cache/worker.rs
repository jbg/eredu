//! Backend-neutral bounded physical worker for cache backing-store tasks.

use std::{
    collections::HashMap,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{mpsc, Arc, Condvar, Mutex},
    thread::{self, JoinHandle},
};

use super::{
    CacheIoAdmission, CacheIoCompletionDisposition, CacheIoExecutionState,
    CacheIoExecutionStateError, CacheIoOperationKey, CacheIoPreparation, CacheIoStartDisposition,
};

enum CacheIoWorkerRequest<Task, Output> {
    Operation {
        key: CacheIoOperationKey,
        task: Box<Task>,
        completion: Arc<CacheIoCompletion<Output>>,
    },
    Stop,
}

#[derive(Debug, Clone)]
enum CacheIoCompletionState<Output> {
    Finished(Result<Output, String>),
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
            CacheIoCompletionState::Cancelled => Err(CacheIoWorkerError::Cancelled { generation }),
        }
    }
}

#[derive(Debug)]
struct CacheIoWorkerShared<Output> {
    in_flight: Mutex<HashMap<CacheIoOperationKey, Arc<CacheIoCompletion<Output>>>>,
    execution: Mutex<CacheIoExecutionState>,
    space_available: Condvar,
}

impl<Output> CacheIoWorkerShared<Output> {
    fn new(capacity: usize) -> Result<Self, CacheIoWorkerError> {
        Ok(Self {
            in_flight: Mutex::new(HashMap::new()),
            execution: Mutex::new(CacheIoExecutionState::new(capacity)?),
            space_available: Condvar::new(),
        })
    }
}

/// Exact completion ownership for one coalesced cache I/O operation.
pub struct CacheIoTicket<Output> {
    /// Exact logical operation identity.
    pub key: CacheIoOperationKey,
    completion: Arc<CacheIoCompletion<Output>>,
    shared: Arc<CacheIoWorkerShared<Output>>,
}

impl<Output> Clone for CacheIoTicket<Output> {
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            completion: Arc::clone(&self.completion),
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
        Arc::ptr_eq(&self.completion, &other.completion)
    }
}

/// Prepared cache I/O that admits physical work only when explicitly enqueued.
pub struct CacheIoSubmission<Task, Output> {
    /// Ticket shared by the operation owner and all exact-key joiners.
    pub ticket: CacheIoTicket<Output>,
    sender: mpsc::Sender<CacheIoWorkerRequest<Task, Output>>,
    shared: Arc<CacheIoWorkerShared<Output>>,
    unsent: Option<CacheIoWorkerRequest<Task, Output>>,
    joined_task: Option<Task>,
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
        self.joined_task.as_mut()
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
                match execution.admit(&self.ticket.key)? {
                    CacheIoAdmission::Admitted => {
                        if self.sender.send(request).is_err() {
                            execution.rollback_admission(&self.ticket.key)?;
                            self.ticket
                                .completion
                                .finish(Err("cache I/O physical worker stopped".into()));
                            self.ticket.completion.release_task_resources();
                        }
                        break;
                    }
                    CacheIoAdmission::AtCapacity => {
                        backpressure = true;
                        execution = match self.shared.space_available.wait(execution) {
                            Ok(execution) => execution,
                            Err(_) => {
                                drop(request);
                                self.ticket.completion.release_task_resources();
                                return Err(CacheIoWorkerError::Poisoned);
                            }
                        };
                    }
                    CacheIoAdmission::Cancelled => {
                        drop(request);
                        self.ticket.completion.release_task_resources();
                        break;
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
    sender: mpsc::Sender<CacheIoWorkerRequest<Task, Output>>,
    handle: Mutex<Option<JoinHandle<()>>>,
    shared: Arc<CacheIoWorkerShared<Output>>,
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
        let (sender, receiver) = mpsc::channel::<CacheIoWorkerRequest<Task, Output>>();
        let shared = Arc::new(CacheIoWorkerShared::new(capacity)?);
        let worker_shared = Arc::clone(&shared);
        let handle = thread::Builder::new()
            .name(thread_name.clone())
            .spawn(move || {
                while let Ok(request) = receiver.recv() {
                    match request {
                        CacheIoWorkerRequest::Operation {
                            key,
                            task,
                            completion,
                        } => {
                            let start = worker_shared
                                .execution
                                .lock()
                                .map_err(|_| CacheIoWorkerError::Poisoned)
                                .and_then(|mut execution| {
                                    execution.begin(&key).map_err(Into::into)
                                });
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
                                    completion.finish(Err(error.to_string()));
                                    completion.release_task_resources();
                                    retire_completion(&worker_shared, &key, &completion);
                                    continue;
                                }
                            }
                            let result = catch_unwind(AssertUnwindSafe(|| execute(*task)))
                                .unwrap_or_else(|_| {
                                    Err("cache I/O physical worker operation panicked".into())
                                });
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
                                if let Ok(output) = result {
                                    discard(output);
                                }
                            } else {
                                completion.finish(result);
                            }
                            completion.release_task_resources();
                            retire_completion(&worker_shared, &key, &completion);
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
        })
    }

    /// Prepares new work or joins an exact operation already owned by the worker.
    pub fn prepare(
        &self,
        key: CacheIoOperationKey,
        task: Task,
    ) -> Result<CacheIoSubmission<Task, Output>, CacheIoWorkerError> {
        let mut execution = self
            .shared
            .execution
            .lock()
            .map_err(|_| CacheIoWorkerError::Poisoned)?;
        let preparation = execution.prepare(key.clone());
        let mut completions = self
            .shared
            .in_flight
            .lock()
            .map_err(|_| CacheIoWorkerError::Poisoned)?;
        if preparation == CacheIoPreparation::Joined {
            let completion = completions
                .get(&key)
                .expect("runtime joined key has an exact completion");
            return Ok(CacheIoSubmission {
                ticket: CacheIoTicket {
                    key,
                    completion: Arc::clone(completion),
                    shared: Arc::clone(&self.shared),
                },
                sender: self.sender.clone(),
                shared: Arc::clone(&self.shared),
                unsent: None,
                joined_task: Some(task),
                joined: true,
            });
        }
        let completion = Arc::new(CacheIoCompletion::default());
        completions.insert(key.clone(), Arc::clone(&completion));
        drop(completions);
        drop(execution);
        let request = CacheIoWorkerRequest::Operation {
            key: key.clone(),
            task: Box::new(task),
            completion: Arc::clone(&completion),
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

    /// Releases exact-key ownership after task resources are safe to drop.
    pub fn retire(&self, ticket: &CacheIoTicket<Output>) {
        retire_completion(&self.shared, &ticket.key, &ticket.completion);
    }
}

impl<Task, Output> Drop for CacheIoWorker<Task, Output> {
    fn drop(&mut self) {
        let _ = self.sender.send(CacheIoWorkerRequest::Stop);
        if let Ok(handle) = self.handle.get_mut() {
            if let Some(handle) = handle.take() {
                let _ = handle.join();
            }
        }
    }
}

fn retire_completion<Output>(
    shared: &CacheIoWorkerShared<Output>,
    key: &CacheIoOperationKey,
    completion: &Arc<CacheIoCompletion<Output>>,
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
                .is_some_and(|current| Arc::ptr_eq(current, completion))
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
            Err(CacheIoWorkerError::OperationFailed(message))
                if message.contains("operation panicked")
        ));
        worker.retire(&ticket);
    }

    #[test]
    fn cancellation_wakes_a_backpressured_submission() {
        let worker =
            Arc::new(CacheIoWorker::new(1, "cache-worker-cancel-test", execute, discard).unwrap());
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let blocker = worker
            .prepare(key(0), Task::Pause(started_tx, release_rx))
            .unwrap();
        let blocker_ticket = blocker.ticket.clone();
        blocker.enqueue().unwrap();
        started_rx.recv().unwrap();

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
}

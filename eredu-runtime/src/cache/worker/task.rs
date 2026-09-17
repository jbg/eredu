//! Paid task/completion shells for the existing exact-key worker.
use super::*;
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError, WorkspaceMetadataFunding},
};
use std::mem::{size_of, size_of_val};

// Every alias drops the actual Arc before its funding alias. In particular the
// final completion allocation and output retire before their account can refund.
pub(super) struct CompletionOwner<Output> {
    inner: Arc<CacheIoCompletion<Output>>,
    funding: Option<WorkspaceMetadataFunding>,
}
impl<Output> Clone for CompletionOwner<Output> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            funding: self.funding.clone(),
        }
    }
}
impl<Output> std::fmt::Debug for CompletionOwner<Output> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheIoCompletionOwner")
            .finish_non_exhaustive()
    }
}
impl<Output> std::ops::Deref for CompletionOwner<Output> {
    type Target = CacheIoCompletion<Output>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<Output> CompletionOwner<Output> {
    fn ordinary() -> Self {
        Self {
            inner: Arc::new(CacheIoCompletion::default()),
            funding: None,
        }
    }
    fn prepared(context: &WorkspaceContext) -> Result<Self, WorkspaceMetadataError> {
        let funding = context
            .metadata_funding()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        Ok(Self {
            inner: context.metadata_arc(CacheIoCompletion::default())?,
            funding: Some(funding),
        })
    }
    pub(super) fn is_prepared(&self) -> bool {
        self.funding.is_some()
    }
    pub(super) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

pub(super) enum TaskInput<Task, Output> {
    Ordinary(Task),
    Prepared {
        task: Box<Option<Task>>,
        completion: CompletionOwner<Output>,
    },
}
impl<Task, Output> TaskInput<Task, Output> {
    pub(super) fn completion(&self) -> CompletionOwner<Output> {
        match self {
            Self::Ordinary(_) => CompletionOwner::ordinary(),
            Self::Prepared { completion, .. } => completion.clone(),
        }
    }
    pub(super) fn task_mut(&mut self) -> &mut Task {
        match self {
            Self::Ordinary(task) => task,
            Self::Prepared { task, .. } => task.as_mut().as_mut().expect("bound prepared task"),
        }
    }
    pub(super) fn into_task(self) -> Box<Option<Task>> {
        match self {
            Self::Ordinary(task) => Box::new(Some(task)),
            Self::Prepared { task, .. } => task,
        }
    }
}

/// One task box and one completion allocation, paid before either allocation.
/// This qualifies shells only: the backend owns payload/source validation,
/// transfer destinations, execution, and result payload qualification.
pub struct PreparedCacheIoTask<Task, Output> {
    input: Option<TaskInput<Task, Output>>,
    key: CacheIoOperationKey,
    source: Arc<CacheIoWorkerShared<Output>>,
}
/// Finite task and completion storage before a backend payload exists.
/// Binding moves one actual payload into the same box; it does not select a
/// worker, change the operation key, allocate, or admit physical work.
pub struct PreparedCacheIoTaskSlot<Task, Output> {
    task: Box<Option<Task>>,
    completion: CompletionOwner<Output>,
    key: CacheIoOperationKey,
    source: Arc<CacheIoWorkerShared<Output>>,
}
impl<Task, Output> PreparedCacheIoTaskSlot<Task, Output> {
    /// Binds exactly one owned backend payload to its already-paid task shell.
    /// Nested source/destination qualification remains the backend's obligation.
    pub fn bind(mut self, task: Task) -> PreparedCacheIoTask<Task, Output> {
        *self.task = Some(task);
        PreparedCacheIoTask {
            input: Some(TaskInput::Prepared {
                task: self.task,
                completion: self.completion,
            }),
            key: self.key,
            source: self.source,
        }
    }
}
/// Unchanged task/source owner returned on foreign-worker or registry refusal.
pub struct CacheIoTaskPreparationError<Task, Output> {
    cause: CacheIoTaskRefusal,
    retained: PreparedCacheIoTask<Task, Output>,
}
/// Exact task preparation refusal. No backend task has run at this point.
#[derive(Debug, thiserror::Error)]
pub enum CacheIoTaskRefusal {
    /// The prepared task belongs to a different actual worker.
    #[error("cache I/O task belongs to a different worker")]
    ForeignWorker,
    /// The same worker rejected its ordinary exact-key preparation transition.
    #[error(transparent)]
    Worker(#[from] CacheIoWorkerError),
}
impl<Task, Output> std::fmt::Debug for PreparedCacheIoTask<Task, Output> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedCacheIoTask")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}
impl<Task, Output> std::fmt::Debug for CacheIoTaskPreparationError<Task, Output> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.cause, f)
    }
}
impl<Task, Output> std::fmt::Display for CacheIoTaskPreparationError<Task, Output> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl<Task: 'static, Output: 'static> std::error::Error
    for CacheIoTaskPreparationError<Task, Output>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl<Task, Output> CacheIoTaskPreparationError<Task, Output> {
    /// Borrow the typed failure without allocating or cloning task payloads.
    pub fn cause(&self) -> &CacheIoTaskRefusal {
        &self.cause
    }
    /// Recover the same one-use task and its source/custody for a later attempt.
    pub fn into_parts(self) -> (CacheIoTaskRefusal, PreparedCacheIoTask<Task, Output>) {
        (self.cause, self.retained)
    }
}
impl<Task, Output> PreparedCacheIoTask<Task, Output> {
    /// Actual task box/shared completion allocation and fixed worker transports.
    /// Nested task/output storage and callback computation are separate costs.
    pub fn control_bytes() -> Option<usize> {
        Self::fixed_bytes()?.checked_add(WorkspaceContext::metadata_arc_bytes::<
            CacheIoCompletion<Output>,
        >()?)
    }
    fn fixed_bytes() -> Option<usize> {
        let frames = [
            size_of::<Option<Task>>(), size_of::<Box<Option<Task>>>(), size_of::<Self>(),
            size_of::<PreparedCacheIoTaskSlot<Task, Output>>(),
            size_of::<(PreparedCacheIoTaskSlot<Task, Output>, Task)>(),
            size_of::<Result<PreparedCacheIoTaskSlot<Task, Output>, Error>>(),
            size_of::<TaskInput<Task, Output>>(), size_of::<CompletionOwner<Output>>(),
            size_of::<CacheIoTaskPreparationError<Task, Output>>(), size_of::<CacheIoTaskRefusal>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Result<CacheIoSubmission<Task, Output>, CacheIoTaskPreparationError<Task, Output>>>(),
            size_of::<CacheIoSubmission<Task, Output>>(), size_of::<CacheIoWorkerRequest<Task, Output>>(),
            size_of::<CacheIoTicket<Output>>(), size_of::<CacheIoSubmissionOutcome>(),
            size_of::<Result<CacheIoSubmissionOutcome, CacheIoWorkerError>>(),
            size_of::<Result<CacheIoSubmission<Task, Output>, CacheIoWorkerError>>(),
            size_of::<Result<Output, String>>(), size_of::<Result<Result<Output, String>, Box<dyn std::any::Any + Send>>>(),
            size_of::<CacheIoCompletionState<Output>>(),
            size_of::<std::sync::MutexGuard<'_, CacheIoExecutionState>>(),
            size_of::<std::sync::MutexGuard<'_, CacheRecordTable<CacheIoOperationKey, CompletionOwner<Output>>>>(),
            size_of::<std::sync::MutexGuard<'_, Option<CacheIoCompletionState<Output>>>>(),
            size_of::<(&WorkspaceContext, CacheIoOperationKey)>(),
            CacheRecordTable::<CacheIoOperationKey, CompletionOwner<Output>>::mutation_control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
impl<Task: Send + 'static, Output: Clone + Send + 'static> PreparedCacheIoTask<Task, Output> {
    /// Uses the same exact-key join/admission worker after checking source and
    /// installed finite queue/registry storage. Refusal preserves this owner.
    pub fn prepare(
        mut self,
        worker: &CacheIoWorker<Task, Output>,
    ) -> Result<CacheIoSubmission<Task, Output>, CacheIoTaskPreparationError<Task, Output>> {
        let result = if Arc::ptr_eq(&self.source, &worker.shared) {
            worker
                .prepare_input(self.key.clone(), &mut self.input, true)
                .map_err(CacheIoTaskRefusal::Worker)
        } else {
            Err(CacheIoTaskRefusal::ForeignWorker)
        };
        result.map_err(|cause| CacheIoTaskPreparationError {
            cause,
            retained: self,
        })
    }
}
impl<Task, Output> CacheIoWorker<Task, Output> {
    /// Prepares a shell for an already-owned backend payload. It does not admit
    /// the operation, clone its source, start I/O, or qualify nested storage.
    pub fn prepare_task(
        &self,
        key: CacheIoOperationKey,
        task: Task,
        context: &WorkspaceContext,
    ) -> Result<PreparedCacheIoTask<Task, Output>, Error> {
        Ok(self.prepare_task_slot(key, context)?.bind(task))
    }

    /// Preallocates the exact task box and completion for this worker and key.
    /// No task is executable until the returned one-use slot receives its payload.
    pub fn prepare_task_slot(
        &self,
        key: CacheIoOperationKey,
        context: &WorkspaceContext,
    ) -> Result<PreparedCacheIoTaskSlot<Task, Output>, Error> {
        context.charge_metadata(
            PreparedCacheIoTask::<Task, Output>::fixed_bytes()
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let completion = CompletionOwner::prepared(context)?;
        Ok(PreparedCacheIoTaskSlot {
            task: Box::new(None),
            completion,
            key,
            source: Arc::clone(&self.shared),
        })
    }
}

/// Allocation-free result inspection. An ordinary task's existing string is
/// borrowed; infrastructure failures and cancellation retain their typed form.
#[derive(Debug)]
pub enum CacheIoBorrowedError<'a> {
    /// Existing ordinary task error, borrowed from the exact completion.
    OperationFailed(&'a str),
    /// Fixed worker state failure.
    Execution(CacheIoExecutionStateError),
    /// Logical cancellation for this exact generation.
    Cancelled { generation: u64 },
    /// Completion synchronization was poisoned.
    Poisoned,
}
impl<Output> CacheIoTicket<Output> {
    /// Fixed controls for waiting on the actual task-resource retirement signal.
    /// The completion/output remains owned by its real tickets after this wait.
    pub fn task_retirement_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<&Self>(),
            size_of::<std::sync::MutexGuard<'_, bool>>(),
            size_of::<std::sync::LockResult<std::sync::MutexGuard<'_, bool>>>(),
            size_of::<Result<(), CacheIoWorkerError>>(),
            size_of::<CacheIoWorkerError>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }

    /// Fixed inspection frames for one actual callback type/result. This does
    /// not price work or escaped payloads created by the callback itself.
    pub fn result_inspection_control_bytes<R, F>() -> Option<usize> {
        let frames = [
            size_of::<F>(),
            size_of::<R>(),
            size_of::<Self>(),
            size_of::<Result<&Output, CacheIoBorrowedError<'_>>>(),
            size_of::<CacheIoBorrowedError<'_>>(),
            size_of::<std::sync::MutexGuard<'_, Option<CacheIoCompletionState<Output>>>>(),
            size_of::<
                std::sync::LockResult<
                    std::sync::MutexGuard<'_, Option<CacheIoCompletionState<Output>>>,
                >,
            >(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Waits and lends the exact output/error without cloning either payload.
    /// The callback must not reenter this completion. References cannot escape;
    /// independently owned result aliases must retain their own payload custody.
    pub fn with_result<R>(
        &self,
        callback: impl FnOnce(Result<&Output, CacheIoBorrowedError<'_>>) -> R,
    ) -> R {
        let mut state = match self.completion.state.lock() {
            Ok(v) => v,
            Err(_) => return callback(Err(CacheIoBorrowedError::Poisoned)),
        };
        while state.is_none() {
            state = match self.completion.ready.wait(state) {
                Ok(v) => v,
                Err(_) => return callback(Err(CacheIoBorrowedError::Poisoned)),
            };
        }
        callback(
            match state.as_ref().expect("completion state was awaited") {
                CacheIoCompletionState::Finished(Ok(output)) => Ok(output),
                CacheIoCompletionState::Finished(Err(cause)) => {
                    Err(CacheIoBorrowedError::OperationFailed(cause))
                }
                CacheIoCompletionState::WorkerFailure(cause) => {
                    Err(CacheIoBorrowedError::Execution(*cause))
                }
                CacheIoCompletionState::Cancelled => Err(CacheIoBorrowedError::Cancelled {
                    generation: self.key.generation,
                }),
            },
        )
    }
}

#[cfg(test)]
#[path = "task/tests.rs"]
mod tests;

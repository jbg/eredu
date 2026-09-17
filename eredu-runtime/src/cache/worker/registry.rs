//! Finite metadata for the same operation and exact-completion registries.
use super::*;
use crate::cache::PreparedCacheTable;
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};
use std::mem::{size_of, size_of_val};

type Completions<Output> = CacheRecordTable<CacheIoOperationKey, CompletionOwner<Output>>;

/// Paid empty destinations for the actual worker's two authoritative registries.
/// This does not qualify task/result, queue-channel or thread allocations and
/// grants no backend I/O, source permission or transfer capacity.
pub struct PreparedCacheIoRegistry<Output> {
    execution: Option<CacheIoExecutionState>,
    completions: Option<Completions<Output>>,
    source: Arc<CacheIoWorkerShared<Output>>,
}

/// Prior empty storage returned after installation. Keep this value until all
/// caller-owned source/manager loans have ended before retiring its custody.
pub struct RetiredCacheIoRegistry<Output> {
    _completions: Completions<Output>,
    _execution: CacheIoExecutionState,
}

/// Exact reason the prepared registry could not replace its worker's storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CacheIoRegistryRefusal {
    /// The destination was prepared from a different actual worker.
    #[error("cache I/O registry belongs to a different worker")]
    ForeignWorker,
    /// Prepared, queued or active work still owns an existing registry entry.
    #[error("cache I/O worker still owns operation resources")]
    Busy,
    /// Shutdown has already begun.
    #[error("cache I/O worker is stopping")]
    Stopped,
    /// An exact worker lock was poisoned.
    #[error("cache I/O registry synchronization is poisoned")]
    Poisoned,
}

/// An unchanged destination and its custody, returned after atomic refusal.
pub struct CacheIoRegistryInstallationError<Output> {
    cause: CacheIoRegistryRefusal,
    retained: PreparedCacheIoRegistry<Output>,
}
impl<Output> CacheIoRegistryInstallationError<Output> {
    /// Allocation-free typed refusal.
    pub fn cause(&self) -> CacheIoRegistryRefusal {
        self.cause
    }
    /// Recovers the unchanged paid destination for a later idle installation.
    pub fn into_parts(self) -> (CacheIoRegistryRefusal, PreparedCacheIoRegistry<Output>) {
        (self.cause, self.retained)
    }
}
impl<Output> std::fmt::Debug for PreparedCacheIoRegistry<Output> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedCacheIoRegistry")
            .finish_non_exhaustive()
    }
}
impl<Output> std::fmt::Debug for RetiredCacheIoRegistry<Output> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetiredCacheIoRegistry")
            .finish_non_exhaustive()
    }
}
impl<Output> std::fmt::Debug for CacheIoRegistryInstallationError<Output> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.cause, f)
    }
}
impl<Output> std::fmt::Display for CacheIoRegistryInstallationError<Output> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl<Output: 'static> std::error::Error for CacheIoRegistryInstallationError<Output> {}

impl<Output> PreparedCacheIoRegistry<Output> {
    /// Exact destination and fixed constructor/installation controls. Maximum
    /// counts all retained operations, including prepared but unqueued tasks.
    pub fn control_bytes(maximum: usize) -> Option<usize> {
        Self::fixed_bytes()?
            .checked_add(CacheIoExecutionState::prepared_bytes(maximum)?)?
            .checked_add(PreparedCacheTable::<
                CacheIoOperationKey,
                CompletionOwner<Output>,
            >::control_bytes(maximum)?)
    }
    fn fixed_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<RetiredCacheIoRegistry<Output>>(),
            size_of::<CacheIoRegistryInstallationError<Output>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<
                Result<RetiredCacheIoRegistry<Output>, CacheIoRegistryInstallationError<Output>>,
            >(),
            size_of::<CacheIoRegistryRefusal>(),
            size_of::<Result<usize, CacheIoRegistryRefusal>>(),
            size_of::<(&WorkspaceContext, usize, usize)>(),
            size_of::<std::sync::MutexGuard<'_, CacheIoExecutionState>>(),
            size_of::<std::sync::MutexGuard<'_, Completions<Output>>>(),
            size_of::<std::sync::TryLockError<std::sync::MutexGuard<'_, CacheIoExecutionState>>>(),
            size_of::<std::sync::TryLockError<std::sync::MutexGuard<'_, Completions<Output>>>>(),
            size_of::<Arc<CacheIoWorkerShared<Output>>>(),
            WorkspaceContext::metadata_source_bytes::<CacheIoRegistryRefusal>()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    /// Atomically installs both empty tables only on their exact idle worker.
    /// A failed attempt changes neither live table and retains the destination.
    pub fn install<Task>(
        mut self,
        worker: &CacheIoWorker<Task, Output>,
    ) -> Result<RetiredCacheIoRegistry<Output>, CacheIoRegistryInstallationError<Output>> {
        let result = (|| {
            if !Arc::ptr_eq(&self.source, &worker.shared) {
                return Err(CacheIoRegistryRefusal::ForeignWorker);
            }
            if worker.shared.stopping.load(Ordering::Acquire) {
                return Err(CacheIoRegistryRefusal::Stopped);
            }
            let mut execution = worker.shared.execution.try_lock().map_err(lock_refusal)?;
            let mut completions = worker.shared.in_flight.try_lock().map_err(lock_refusal)?;
            if !execution.is_empty()
                || !completions.is_empty()
                || worker.shared.active_payload.load(Ordering::Acquire)
            {
                return Err(CacheIoRegistryRefusal::Busy);
            }
            let mut replacement = self.execution.take().expect("one registry installation");
            replacement.preserve_history(&execution);
            debug_assert_eq!(replacement.capacity(), execution.capacity());
            let prior_execution = std::mem::replace(&mut *execution, replacement);
            let prior_completions = std::mem::replace(
                &mut *completions,
                self.completions.take().expect("one registry installation"),
            );
            Ok(RetiredCacheIoRegistry {
                _completions: prior_completions,
                _execution: prior_execution,
            })
        })();
        result.map_err(|cause| CacheIoRegistryInstallationError {
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
    /// Prepares exact registry storage without replacing it or admitting work.
    /// Payload/thread/channel source producers remain independently required.
    pub fn prepare_registry(
        &self,
        maximum: usize,
        context: &WorkspaceContext,
    ) -> Result<PreparedCacheIoRegistry<Output>, Error> {
        context.charge_metadata(
            PreparedCacheIoRegistry::<Output>::fixed_bytes()
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let capacity = self
            .shared
            .execution
            .try_lock()
            .map(|state| state.capacity())
            .map_err(|cause| context.metadata_source(lock_refusal(cause)))?;
        let execution = CacheIoExecutionState::prepared(capacity, maximum, context)?;
        let storage = PreparedCacheTable::prepare(maximum, context)?;
        let mut completions = CacheRecordTable::new();
        drop(
            completions
                .install(storage)
                .map_err(|_| WorkspaceMetadataError::Unqualified)?,
        );
        Ok(PreparedCacheIoRegistry {
            execution: Some(execution),
            completions: Some(completions),
            source: Arc::clone(&self.shared),
        })
    }
}

#[cfg(test)]
#[path = "registry/tests.rs"]
mod tests;

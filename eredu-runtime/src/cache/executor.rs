//! Backend-neutral bounded admission for cache backing-store operations.

use super::{CacheRecordTable, CacheTableCapacityError, PreparedCacheTable};
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError},
};

use super::CacheIoOperationKey;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum OperationPhase {
    Prepared,
    Queued,
    InFlight,
    CancelledPrepared,
    CancelledQueued,
    CancelledInFlight,
    Completed,
    CompletedCancelled,
}

/// Result of preparing an exact cache I/O key.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CacheIoPreparation {
    /// The backend must allocate resources for a new operation.
    New,
    /// An existing exact operation owns the result and should be joined.
    Joined,
}

/// Result of requesting bounded queue admission for prepared cache I/O.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CacheIoAdmission {
    /// The physical executor may enqueue this operation.
    Admitted,
    /// Queue capacity must become available before retrying admission.
    AtCapacity,
    /// Cancellation won before physical enqueue.
    Cancelled,
}

/// Action a physical executor takes after receiving one admitted operation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CacheIoStartDisposition {
    /// Execute the backend task and report its exact completion.
    Execute,
    /// Drop a queued task whose logical operation was cancelled or retired.
    Discard,
}

/// Publication action after a physically started operation completes.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CacheIoCompletionDisposition {
    /// Publish the backend result to exact completion waiters.
    Publish,
    /// Cancellation won; discard the backend result and retained resources.
    Discard,
}

/// Canonical admission and cancellation state for a bounded cache I/O worker.
///
/// Backends own task payloads, worker threads, files, buffers, and completion
/// objects. This state owns exact-key coalescing, bounded admission, queued and
/// in-flight cancellation, and peak queue accounting.
#[derive(Debug)]
pub struct CacheIoExecutionState {
    capacity: usize,
    queued: usize,
    peak_queued: usize,
    operations: CacheRecordTable<CacheIoOperationKey, OperationPhase>,
    prepared: bool,
}

impl CacheIoExecutionState {
    /// Creates an empty state with finite nonzero queue capacity.
    pub fn new(capacity: usize) -> Result<Self, CacheIoExecutionStateError> {
        if capacity == 0 {
            return Err(CacheIoExecutionStateError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            queued: 0,
            peak_queued: 0,
            operations: CacheRecordTable::new(),
            prepared: false,
        })
    }

    /// Registers an exact key for the ordinary growing worker. Callers of an
    /// installed finite registry use [`Self::try_prepare`] to retain refusal.
    pub fn prepare(&mut self, key: CacheIoOperationKey) -> CacheIoPreparation {
        self.try_prepare(key)
            .expect("ordinary cache I/O registry can grow")
    }

    /// Registers or joins the same exact key. A finite registry refuses a new
    /// key at its explicit retained-operation limit without changing any state.
    pub fn try_prepare(
        &mut self,
        key: CacheIoOperationKey,
    ) -> Result<CacheIoPreparation, CacheIoExecutionStateError> {
        if self.operations.contains_key(&key) {
            return Ok(CacheIoPreparation::Joined);
        }
        if self.prepared {
            self.operations
                .insert_prepared(key, OperationPhase::Prepared)
                .map_err(|(cause, _, _)| CacheIoExecutionStateError::RegistryCapacity(cause))?;
        } else {
            self.operations.insert(key, OperationPhase::Prepared);
        }
        Ok(CacheIoPreparation::New)
    }

    pub(super) fn prepared_bytes(maximum: usize) -> Option<usize> {
        PreparedCacheTable::<CacheIoOperationKey, OperationPhase>::control_bytes(maximum)?
            .checked_add(std::mem::size_of::<(Self, Result<Self, Error>, usize, usize)>())
    }
    pub(super) fn prepared(
        capacity: usize,
        maximum: usize,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        context
            .charge_metadata(std::mem::size_of::<(Self, Result<Self, Error>, usize, usize)>())?;
        if capacity == 0 {
            return Err(context.metadata_source(CacheIoExecutionStateError::ZeroCapacity));
        }
        let storage = PreparedCacheTable::prepare(maximum, context)?;
        let mut operations = CacheRecordTable::new();
        drop(
            operations
                .install(storage)
                .map_err(|_| WorkspaceMetadataError::Unqualified)?,
        );
        Ok(Self {
            capacity,
            queued: 0,
            peak_queued: 0,
            operations,
            prepared: true,
        })
    }
    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }
    pub(super) fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
    pub(super) fn is_prepared(&self) -> bool {
        self.prepared
    }
    pub(super) fn preserve_history(&mut self, prior: &Self) {
        debug_assert_eq!(self.capacity, prior.capacity);
        debug_assert!(prior.is_empty());
        self.peak_queued = prior.peak_queued;
    }

    /// Attempts to admit one previously prepared operation.
    pub fn admit(
        &mut self,
        key: &CacheIoOperationKey,
    ) -> Result<CacheIoAdmission, CacheIoExecutionStateError> {
        let phase = self
            .operations
            .get(key)
            .copied()
            .ok_or(CacheIoExecutionStateError::UnknownOperation)?;
        match phase {
            OperationPhase::Prepared => {
                if self.queued == self.capacity {
                    return Ok(CacheIoAdmission::AtCapacity);
                }
                self.queued += 1;
                self.peak_queued = self.peak_queued.max(self.queued);
                *self
                    .operations
                    .get_mut(key)
                    .expect("existing operation phase") = OperationPhase::Queued;
                Ok(CacheIoAdmission::Admitted)
            }
            OperationPhase::CancelledPrepared => Ok(CacheIoAdmission::Cancelled),
            _ => Err(CacheIoExecutionStateError::InvalidAdmission),
        }
    }

    /// Rolls back physical enqueue failure without losing exact-key ownership.
    pub fn rollback_admission(
        &mut self,
        key: &CacheIoOperationKey,
    ) -> Result<(), CacheIoExecutionStateError> {
        match self.operations.get(key) {
            Some(OperationPhase::Queued) => {
                self.queued -= 1;
                *self
                    .operations
                    .get_mut(key)
                    .expect("existing operation phase") = OperationPhase::Prepared;
                Ok(())
            }
            Some(OperationPhase::CancelledQueued) => {
                self.queued -= 1;
                *self
                    .operations
                    .get_mut(key)
                    .expect("existing operation phase") = OperationPhase::CancelledPrepared;
                Ok(())
            }
            _ => Err(CacheIoExecutionStateError::InvalidAdmissionRollback),
        }
    }

    /// Begins one operation received by the physical executor.
    pub fn begin(
        &mut self,
        key: &CacheIoOperationKey,
    ) -> Result<CacheIoStartDisposition, CacheIoExecutionStateError> {
        let Some(phase) = self.operations.get(key).copied() else {
            // A cancelled queued ticket may be retired by its waiter before
            // the physical executor drains the opaque task.
            return Ok(CacheIoStartDisposition::Discard);
        };
        match phase {
            OperationPhase::Queued => {
                self.queued -= 1;
                *self
                    .operations
                    .get_mut(key)
                    .expect("existing operation phase") = OperationPhase::InFlight;
                Ok(CacheIoStartDisposition::Execute)
            }
            OperationPhase::CancelledQueued => {
                self.queued -= 1;
                *self
                    .operations
                    .get_mut(key)
                    .expect("existing operation phase") = OperationPhase::CompletedCancelled;
                Ok(CacheIoStartDisposition::Discard)
            }
            _ => Err(CacheIoExecutionStateError::InvalidStart),
        }
    }

    /// Cancels prepared, queued, or physically executing work exactly once.
    pub fn cancel(&mut self, key: &CacheIoOperationKey) -> bool {
        let Some(phase) = self.operations.get(key).copied() else {
            return false;
        };
        let cancelled = match phase {
            OperationPhase::Prepared => OperationPhase::CancelledPrepared,
            OperationPhase::Queued => OperationPhase::CancelledQueued,
            OperationPhase::InFlight => OperationPhase::CancelledInFlight,
            _ => return false,
        };
        *self
            .operations
            .get_mut(key)
            .expect("existing operation phase") = cancelled;
        true
    }

    /// Resolves one exact physically executed operation.
    pub fn complete(
        &mut self,
        key: &CacheIoOperationKey,
    ) -> Result<CacheIoCompletionDisposition, CacheIoExecutionStateError> {
        match self.operations.get(key) {
            Some(OperationPhase::InFlight) => {
                *self
                    .operations
                    .get_mut(key)
                    .expect("existing operation phase") = OperationPhase::Completed;
                Ok(CacheIoCompletionDisposition::Publish)
            }
            Some(OperationPhase::CancelledInFlight) => {
                *self
                    .operations
                    .get_mut(key)
                    .expect("existing operation phase") = OperationPhase::CompletedCancelled;
                Ok(CacheIoCompletionDisposition::Discard)
            }
            _ => Err(CacheIoExecutionStateError::InvalidCompletion),
        }
    }

    /// Releases exact-key ownership after task resources are safe to drop.
    pub fn retire(
        &mut self,
        key: &CacheIoOperationKey,
    ) -> Result<bool, CacheIoExecutionStateError> {
        let Some(phase) = self.operations.get(key).copied() else {
            return Ok(false);
        };
        if matches!(
            phase,
            OperationPhase::Queued
                | OperationPhase::InFlight
                | OperationPhase::CancelledQueued
                | OperationPhase::CancelledInFlight
        ) {
            return Err(CacheIoExecutionStateError::OperationStillOwned);
        }
        self.operations.remove(key);
        Ok(true)
    }

    /// Current physically queued operation count.
    pub const fn queued(&self) -> usize {
        self.queued
    }

    /// Largest physically queued operation count.
    pub const fn peak_queued(&self) -> usize {
        self.peak_queued
    }
}

/// Invalid cache I/O executor lifecycle use.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum CacheIoExecutionStateError {
    /// The worker's physical FIFO has no shutdown slot after checked sizing.
    #[error("cache I/O physical queue size overflow")]
    QueueSizeOverflow,
    /// The actual finite FIFO refused a message despite logical admission.
    #[error("cache I/O physical queue capacity is exhausted")]
    QueueCapacity,
    /// The physical worker no longer accepts submitted resources.
    #[error("cache I/O physical worker stopped")]
    WorkerStopped,
    /// A shared coordination lock was poisoned.
    #[error("cache I/O coordination lock is poisoned")]
    CoordinationPoisoned,
    /// The backend task panicked inside the existing worker boundary.
    #[error("cache I/O physical worker operation panicked")]
    TaskPanicked,
    /// The exact finite retained-operation registry cannot accept a new key.
    #[error(transparent)]
    RegistryCapacity(#[from] CacheTableCapacityError),
    /// Bounded I/O admission requires at least one slot.
    #[error("cache I/O queue capacity must be nonzero")]
    ZeroCapacity,
    /// The exact operation key was never prepared or was already retired.
    #[error("cache I/O operation is unknown or already retired")]
    UnknownOperation,
    /// Admission was attempted from a non-prepared phase.
    #[error("cache I/O operation cannot be admitted from its current phase")]
    InvalidAdmission,
    /// Physical enqueue rollback did not own an admitted operation.
    #[error("cache I/O admission cannot be rolled back from its current phase")]
    InvalidAdmissionRollback,
    /// The physical executor received an operation from an invalid phase.
    #[error("cache I/O operation cannot start from its current phase")]
    InvalidStart,
    /// Completion did not correspond to a physically executing operation.
    #[error("cache I/O operation cannot complete from its current phase")]
    InvalidCompletion,
    /// Retirement preceded physical dequeue or exact completion.
    #[error("cache I/O operation still owns queued or in-flight resources")]
    OperationStillOwned,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CacheIoOperationKind;
    use eredu_core::cache::{CacheBlockId, CacheRepresentation};

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
    fn exact_keys_coalesce_and_capacity_is_core_owned() {
        let mut state = CacheIoExecutionState::new(1).unwrap();
        let first = key(0);
        let second = key(1);
        assert_eq!(state.prepare(first.clone()), CacheIoPreparation::New);
        assert_eq!(state.prepare(first.clone()), CacheIoPreparation::Joined);
        assert_eq!(state.admit(&first).unwrap(), CacheIoAdmission::Admitted);
        assert_eq!(state.prepare(second.clone()), CacheIoPreparation::New);
        assert_eq!(state.admit(&second).unwrap(), CacheIoAdmission::AtCapacity);
        assert_eq!(
            state.begin(&first).unwrap(),
            CacheIoStartDisposition::Execute
        );
        assert_eq!(state.admit(&second).unwrap(), CacheIoAdmission::Admitted);
        assert_eq!(state.peak_queued(), 1);
    }

    #[test]
    fn queued_and_in_flight_cancellation_preserve_exact_ownership() {
        let mut state = CacheIoExecutionState::new(2).unwrap();
        let queued = key(0);
        state.prepare(queued.clone());
        state.admit(&queued).unwrap();
        assert!(state.cancel(&queued));
        assert_eq!(
            state.retire(&queued),
            Err(CacheIoExecutionStateError::OperationStillOwned)
        );
        assert_eq!(
            state.begin(&queued).unwrap(),
            CacheIoStartDisposition::Discard
        );
        assert!(!state.cancel(&queued));
        assert!(state.retire(&queued).unwrap());

        let active = key(1);
        state.prepare(active.clone());
        state.admit(&active).unwrap();
        assert_eq!(
            state.begin(&active).unwrap(),
            CacheIoStartDisposition::Execute
        );
        assert!(state.cancel(&active));
        assert_eq!(
            state.complete(&active).unwrap(),
            CacheIoCompletionDisposition::Discard
        );
        assert!(state.retire(&active).unwrap());
    }

    #[test]
    fn failed_physical_enqueue_rolls_back_capacity_for_recovery() {
        let mut state = CacheIoExecutionState::new(1).unwrap();
        let operation = key(0);
        state.prepare(operation.clone());
        state.admit(&operation).unwrap();
        state.rollback_admission(&operation).unwrap();
        assert_eq!(state.queued(), 0);
        assert_eq!(state.admit(&operation).unwrap(), CacheIoAdmission::Admitted);
    }
}

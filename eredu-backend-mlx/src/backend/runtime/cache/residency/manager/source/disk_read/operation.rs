//! Exact shared DiskReady -> DiskReading -> HostBacked publication.
use super::*;
use eredu_runtime::cache::{CacheIoBorrowedError, CacheIoTaskRefusal, CacheIoWorkerError};
type Finished = Arc<Mutex<Option<CompletedDiskRead>>>;
type NativeFailure = Arc<OnceLock<DiskReadFinishFailure>>;
#[path = "operation/promotion.rs"]
mod promotion;
pub(crate) use promotion::ReadCacheHostSource;

pub(super) struct CompletionSlots {
    finished: Finished,
    native_failure: NativeFailure,
}
impl CompletionSlots {
    pub(super) fn prepare(
        context: &WorkspaceContext,
        publication_controls: usize,
    ) -> Result<Self, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        context
            .charge_metadata(
                DiskReadOperation::control_bytes()
                    .and_then(|n| n.checked_add(publication_controls.checked_mul(6)?))
                    .and_then(|n| n.checked_add(size_of::<Self>()))
                    .and_then(|n| n.checked_add(size_of::<(&WorkspaceContext, usize)>()))
                    .and_then(|n| n.checked_add(size_of::<Result<Self, CacheSourceFailure>>()))
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        Ok(Self {
            finished: context
                .metadata_arc(Mutex::new(None))
                .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?,
            native_failure: context
                .metadata_arc(OnceLock::new())
                .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?,
        })
    }
}
pub(crate) struct DiskReadOperation {
    finished: Finished,
    native_failure: NativeFailure,
    task: Option<PreparedCacheIoTask<DiskTask, DiskResult>>,
    submission: Option<RuntimeDiskSubmission>,
    ticket: Option<DiskTicket>,
    output: PreparedDiskReadOutput,
    source: LiveCacheBlockSource,
    location: DiskLocation,
    manager: CacheResidencyManager,
    worker: Arc<DiskWorker>,
    key: CacheIoOperationKey,
    occupancy: DiskReadOccupancy,
    logical_bytes: u64,
    attempted: bool,
    submitted: bool,
    finish_started: bool,
    armed: bool,
    committed: bool,
    promotion_attempted: bool,
    funding: HostMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Source(CacheSourceError),
    #[error(transparent)]
    Task(CacheIoTaskRefusal),
    #[error(transparent)]
    Worker(CacheIoWorkerError),
    #[error(transparent)]
    Policy(CacheResidencyError),
    #[error("prepared disk completion belongs to another task")]
    Completion,
    #[error("prepared disk read Host publication failed")]
    Host,
}
pub(crate) struct DiskReadOperationFailure {
    cause: Cause,
    finished: Finished,
    native_failure: NativeFailure,
    output: PreparedDiskReadOutput,
    funding: HostMetadataFunding,
}
impl std::fmt::Debug for DiskReadOperationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskReadOperationFailure")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl std::fmt::Display for DiskReadOperationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for DiskReadOperationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if matches!(self.cause, Cause::Host) {
            if let Some(cause) = self.native_failure.get() {
                return Some(cause);
            }
        }
        Some(&self.cause)
    }
}
pub(in super::super) fn prepare_operation(
    mut read: PreparedDiskRead,
    manager: &CacheResidencyManager,
    worker: &Arc<DiskWorker>,
    generation: u64,
    context: &WorkspaceContext,
) -> Result<DiskReadOperation, CacheSourceFailure> {
    let fail = |cause| CacheSourceFailure::source(cause, context);
    if !Arc::ptr_eq(&read.body.manager.inner, &manager.inner)
        || !Arc::ptr_eq(&read.worker, worker)
        || read.body.generation != generation
    {
        return Err(fail(CacheSourceError::Identity));
    }
    let funding = context
        .metadata_funding()
        .ok_or_else(|| fail(CacheSourceError::Identity))?;
    if !funding.same_account(&read.body.funding) {
        return Err(fail(CacheSourceError::Identity));
    }
    let slots = read
        .operation
        .take()
        .ok_or_else(|| fail(CacheSourceError::Identity))?;
    let mut operation = DiskReadOperation {
        finished: slots.finished,
        native_failure: slots.native_failure,
        task: None,
        submission: None,
        ticket: None,
        output: read.output.clone(),
        source: read.body.source.clone(),
        location: read.body.location.clone(),
        manager: manager.clone(),
        worker: worker.clone(),
        key: CacheIoOperationKey {
            generation,
            id: read.body.id.clone(),
            kind: CacheIoOperationKind::Read,
        },
        occupancy: read.body.reservation.clone(),
        logical_bytes: read
            .body
            .layout
            .tensor_metadata()
            .iter()
            .try_fold(0u64, |n, (_, _, bytes)| {
                n.checked_add(u64::try_from(*bytes).ok()?)
            })
            .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        attempted: false,
        submitted: false,
        finish_started: false,
        armed: false,
        committed: false,
        promotion_attempted: false,
        funding,
    };
    operation.task = Some(
        read.prepare_task(context)
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?,
    );
    Ok(operation)
}
fn lock_cause<T>(cause: TryLockError<T>) -> CacheSourceError {
    match cause {
        TryLockError::WouldBlock => CacheSourceError::Busy,
        TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
    }
}
impl DiskReadOperation {
    fn failure(&self, cause: Cause) -> DiskReadOperationFailure {
        DiskReadOperationFailure {
            cause,
            finished: self.finished.clone(),
            native_failure: self.native_failure.clone(),
            output: self.output.clone(),
            funding: self.funding.clone(),
        }
    }
    fn validate(&self, state: &CacheManagerState, pending: bool) -> Result<(), Cause> {
        let fail = || Cause::Source(CacheSourceError::Identity);
        if state.generation != self.key.generation
            || self
                .manager
                .inner
                .disk_worker
                .as_ref()
                .is_none_or(|worker| !Arc::ptr_eq(worker, &self.worker))
            || state
                .lifecycle
                .source_pin_count(&self.key.id)
                .map_err(|cause| Cause::Source(cause.into()))?
                == 0
        {
            return Err(fail());
        }
        let record = state.blocks.get(&self.key.id).ok_or_else(fail)?;
        if record.physical.phase()
            != if pending {
                CacheStoragePhase::DiskReading
            } else {
                CacheStoragePhase::DiskReady
            }
            || record.bytes != self.logical_bytes
            || record
                .disk()
                .and_then(|location| location.live_source.as_ref())
                .is_none_or(|source| !source.same_source(&self.source))
        {
            return Err(fail());
        }
        if pending
            && (!record.physical.io_matches(&self.key)
                || record
                    .physical
                    .io()
                    .and_then(|io| io.prepared_read.as_ref())
                    .is_none_or(|owner| !Arc::ptr_eq(&owner.inner, &self.occupancy.inner)))
        {
            return Err(fail());
        }
        Ok(())
    }
    pub(crate) fn submit(&mut self) -> Result<(), DiskReadOperationFailure> {
        self.submit_inner().map_err(|cause| self.failure(cause))
    }
    fn submit_inner(&mut self) -> Result<(), Cause> {
        if self.attempted {
            return Err(Cause::Source(CacheSourceError::Identity));
        }
        self.attempted = true;
        let task = self
            .task
            .take()
            .ok_or(Cause::Source(CacheSourceError::Identity))?;
        let submission = match task.prepare(&self.worker.inner) {
            Ok(value) => value,
            Err(error) => {
                let (cause, task) = error.into_parts();
                self.task = Some(task);
                return Err(Cause::Task(cause));
            }
        };
        if submission.joined {
            self.submission = Some(submission);
            return Err(Cause::Completion);
        }
        self.ticket = Some(DiskTicket {
            inner: submission.ticket.clone(),
        });
        self.submission = Some(submission);
        {
            let mut state = self
                .manager
                .inner
                .state
                .try_lock()
                .map_err(|cause| Cause::Source(lock_cause(cause)))?;
            self.validate(&state, false)?;
            let required = state
                .telemetry
                .report
                .current_host_bytes
                .checked_add(self.occupancy.host_bytes)
                .ok_or(Cause::Source(CacheSourceError::Overflow))?;
            if required > self.manager.options().host_budget_bytes() {
                return Err(Cause::Policy(CacheResidencyError::BudgetExceeded {
                    tier: CacheTier::Host,
                    required,
                    budget: self.manager.options().host_budget_bytes(),
                }));
            }
            reporting::update_report_totals_prepared(&mut state).map_err(Cause::Policy)?;
            state
                .blocks
                .get_mut(&self.key.id)
                .expect("validated disk source")
                .physical
                .begin_read(MlxCacheIoOperation {
                    ticket: self.ticket.as_ref().expect("prepared task").clone(),
                    reserved_host_bytes: Some(self.occupancy.host_bytes),
                    prepared_read: Some(self.occupancy.clone()),
                })
                .expect("same locked stable disk source");
            self.armed = true;
            reporting::update_report_totals_prepared(&mut state).map_err(Cause::Policy)?;
        }
        let submission = self.submission.take().expect("prepared task submission");
        self.submitted = true;
        submission.enqueue().map_err(Cause::Worker)?;
        Ok(())
    }
    pub(crate) fn finish(&mut self) -> Result<(), DiskReadOperationFailure> {
        self.finish_inner().map_err(|cause| self.failure(cause))
    }
    fn finish_inner(&mut self) -> Result<(), Cause> {
        if !self.submitted || !self.armed || self.finish_started {
            return Err(Cause::Source(CacheSourceError::Identity));
        }
        self.finish_started = true;
        let actual = self
            .ticket
            .as_ref()
            .expect("submitted task")
            .inner
            .with_result(inspect_result)?;
        if !Arc::ptr_eq(&actual.inner, &self.output.inner) {
            return Err(Cause::Completion);
        }
        self.ticket
            .as_ref()
            .expect("submitted task")
            .inner
            .wait_for_task_resources().map_err(Cause::Worker)?;
        // Return to the actual creator for freeze/registration before borrowing
        // the manager. Both success and failed prefixes enter their paid owner.
        let completed = match self.output.finish() {
            Ok(value) => value,
            Err(cause) => {
                if self.native_failure.set(cause).is_err() {
                    unreachable!("single finish failure");
                }
                return Err(Cause::Host);
            }
        };
        let mut finished = self
            .finished
            .try_lock()
            .map_err(|cause| Cause::Source(lock_cause(cause)))?;
        *finished = Some(completed);
        let completed = finished.as_ref().expect("retained completed read");
        let host = completed.host().clone();
        let mut state = self
            .manager
            .inner
            .state
            .try_lock()
            .map_err(|cause| Cause::Source(lock_cause(cause)))?;
        self.validate(&state, true)?;
        let mut occupancy = self
            .occupancy
            .inner
            .try_lock()
            .map_err(|cause| Cause::Source(lock_cause(cause)))?;
        reporting::update_report_totals_prepared(&mut state).map_err(Cause::Policy)?;
        let io = state
            .blocks
            .get_mut(&self.key.id)
            .expect("validated pending read")
            .physical
            .finish_read(&self.key, host)
            .expect("same locked pending source");
        if let Err(cause) = reporting::update_report_totals_prepared_replacement(
            &mut state,
            &mut occupancy,
            &self.manager.inner.pool_membership,
        ) {
            let mut original =
                MlxCacheBlockStorage::disk(self.key.id.clone(), self.location.clone());
            original
                .begin_read(io)
                .expect("same exact disk read rollback");
            let retired = std::mem::replace(
                &mut state
                    .blocks
                    .get_mut(&self.key.id)
                    .expect("same source")
                    .physical,
                original,
            );
            let _ = reporting::update_report_totals_prepared(&mut state);
            drop(state);
            drop(occupancy);
            drop(retired);
            return Err(Cause::Policy(cause));
        }
        self.armed = false;
        self.committed = true;
        drop(state);
        drop(occupancy);
        drop(io);
        Ok(())
    }
    pub(crate) fn completed_source_pin_count(&self) -> usize {
        usize::from(self.committed)
    }
    fn control_bytes() -> Option<usize> {
        type StateLoan<'a> = MutexGuard<'a, CacheManagerState>;
        type PoolLoan<'a> = MutexGuard<'a, CachePoolReservation>;
        let frames = [
            reporting::report_query_control_bytes()?,
            RuntimeCacheIoTicket::<DiskResult>::task_retirement_control_bytes()?,
            size_of::<Self>(),
            size_of::<Cause>(),
            size_of::<DiskReadOperationFailure>(),
            size_of::<Result<Self, CacheSourceFailure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<Result<(), DiskReadOperationFailure>>(),
            size_of::<Result<CompletedDiskRead, DiskReadFinishFailure>>(),
            size_of::<Result<(), DiskReadFinishFailure>>(),
            size_of::<Result<PreparedDiskReadOutput, Cause>>(),
            size_of::<Finished>(),
            size_of::<NativeFailure>(),
            WorkspaceContext::metadata_arc_bytes::<Mutex<Option<CompletedDiskRead>>>()?,
            WorkspaceContext::metadata_arc_bytes::<OnceLock<DiskReadFinishFailure>>()?,
            size_of::<PreparedCacheIoTask<DiskTask, DiskResult>>(),
            size_of::<RuntimeDiskSubmission>(),
            size_of::<DiskTicket>(),
            size_of::<CacheIoOperationKey>(),
            size_of::<MlxCacheIoOperation>(),
            size_of::<MlxCacheBlockStorage>(),
            size_of::<HostCacheBlock>(),
            size_of::<DiskLocation>(),
            size_of::<Result<StateLoan<'_>, TryLockError<StateLoan<'_>>>>(),
            size_of::<Result<PoolLoan<'_>, TryLockError<PoolLoan<'_>>>>(),
            size_of::<MutexGuard<'_, Option<CompletedDiskRead>>>(),
            size_of::<(u64, bool, Option<MlxCacheIoOperation>)>(),
            size_of::<(&mut Self, &CacheManagerState, bool)>(),
            size_of::<(
                PreparedDiskRead,
                &CacheResidencyManager,
                &Arc<DiskWorker>,
                u64,
                &WorkspaceContext,
            )>(),
            size_of::<[(&[usize], StoredDtype, usize); 2]>(),
            RuntimeCacheIoTicket::<DiskResult>::result_inspection_control_bytes::<
                Result<PreparedDiskReadOutput, Cause>,
                fn(
                    Result<&DiskResult, CacheIoBorrowedError<'_>>,
                ) -> Result<PreparedDiskReadOutput, Cause>,
            >()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
fn inspect_result(
    result: Result<&DiskResult, CacheIoBorrowedError<'_>>,
) -> Result<PreparedDiskReadOutput, Cause> {
    match result {
        Ok(DiskResult::PreparedRead(output)) if output.result().is_some() => Ok(output.clone()),
        Ok(_) | Err(CacheIoBorrowedError::OperationFailed(_)) => Err(Cause::Completion),
        Err(CacheIoBorrowedError::Execution(cause)) => Err(Cause::Worker(cause.into())),
        Err(CacheIoBorrowedError::Cancelled { generation }) => {
            Err(Cause::Worker(CacheIoWorkerError::Cancelled { generation }))
        }
        Err(CacheIoBorrowedError::Poisoned) => Err(Cause::Source(CacheSourceError::Poisoned)),
    }
}
impl Drop for DiskReadOperation {
    fn drop(&mut self) {
        if self.armed {
            if let Some(ticket) = &self.ticket {
                ticket.cancel();
            }
            let mut state = self
                .manager
                .inner
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let own = state
                .blocks
                .get(&self.key.id)
                .and_then(|record| record.physical.io())
                .and_then(|io| io.prepared_read.as_ref())
                .is_some_and(|owner| Arc::ptr_eq(&owner.inner, &self.occupancy.inner));
            let removed = own
                .then(|| {
                    state.blocks.get_mut(&self.key.id).and_then(|record| {
                        record
                            .physical
                            .io_matches(&self.key)
                            .then(|| record.physical.fail_io(&self.key).ok())
                            .flatten()
                    })
                })
                .flatten();
            let _ = reporting::update_report_totals_prepared(&mut state);
            drop(state);
            drop(removed);
        }
        // Task/output/creator-failure and physical occupancy retire only now,
        // outside the manager lock. Cancellation never implies I/O completion.
    }
}

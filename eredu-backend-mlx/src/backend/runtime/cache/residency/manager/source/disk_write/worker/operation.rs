//! Exact pending-write transition over the installed shared disk worker.
use super::*;
use eredu_runtime::cache::{CacheIoBorrowedError, CacheIoTaskRefusal, CacheIoWorkerError};

/// A single actual task, canonical pending row, and completion. The caller keeps
/// this owner through success or failure. Drop cancels only its own pending row.
pub(crate) struct DiskWriteOperation {
    task: Option<PreparedCacheIoTask<DiskTask, DiskResult>>,
    submission: Option<RuntimeDiskSubmission>,
    ticket: Option<DiskTicket>,
    output: PreparedDiskWriteOutput,
    host: Option<HostCacheBlock>,
    occupancy: DiskWriteOccupancy,
    manager: CacheResidencyManager,
    worker: Arc<DiskWorker>,
    key: CacheIoOperationKey,
    logical_bytes: u64,
    source_pins: usize,
    attempted: bool,
    finish_started: bool,
    armed: bool,
    submitted: bool,
    committed: bool,
    funding: HostMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Source(CacheSourceError),
    #[error(transparent)]
    SourceProof(Exception),
    #[error(transparent)]
    Task(CacheIoTaskRefusal),
    #[error(transparent)]
    Worker(CacheIoWorkerError),
    #[error(transparent)]
    Policy(CacheResidencyError),
    #[error("prepared disk write failed; the output retains its exact cause")]
    Write,
    #[error("prepared disk completion belongs to a different task")]
    Completion,
}
/// Fixed error transport sharing the real asynchronous output and its H.
/// Failed file/native prefixes remain inside that immutable completion owner.
pub(crate) struct DiskWriteOperationFailure {
    cause: Cause,
    output: PreparedDiskWriteOutput,
    funding: HostMetadataFunding,
}
impl std::fmt::Debug for DiskWriteOperationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskWriteOperationFailure")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
impl std::fmt::Display for DiskWriteOperationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for DiskWriteOperationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if matches!(self.cause, Cause::Write) {
            if let Some(Err(cause)) = self.output.result() {
                return Some(cause);
            }
        }
        Some(&self.cause)
    }
}
impl InstalledDiskWorker {
    /// Moves the source into its cold task box after validating the exact
    /// worker and prepaid account. The source loan has ended before this call.
    pub(crate) fn prepare_write(
        &self,
        write: PreparedDiskWrite,
        context: &WorkspaceContext,
    ) -> Result<DiskWriteOperation, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        let funding = context.metadata_funding().ok_or_else(|| {
            CacheSourceFailure::metadata(WorkspaceMetadataError::Unqualified.into(), context)
        })?;
        if !Arc::ptr_eq(&self.manager.inner, &write.body.manager.inner)
            || !Arc::ptr_eq(&self.worker, &write.worker)
            || self.generation != write.body.generation
        {
            return Err(fail(CacheSourceError::Identity));
        }
        if !funding.same_account(&write.body.funding) {
            return Err(fail(CacheSourceError::Identity));
        }
        let mut operation = DiskWriteOperation {
            task: None,
            submission: None,
            ticket: None,
            output: write.output.clone(),
            host: write.body.host.clone(),
            occupancy: write.body.transfer.clone(),
            manager: self.manager.clone(),
            worker: self.worker.clone(),
            key: CacheIoOperationKey {
                generation: self.generation,
                id: write.body.id.clone(),
                kind: CacheIoOperationKind::Write,
            },
            source_pins: write.body.source_pins,
            logical_bytes: write
                .descriptors_bytes()
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            attempted: false,
            finish_started: false,
            armed: false,
            submitted: false,
            committed: false,
            funding,
        };
        operation.task = Some(
            write
                .prepare_task(context)
                .map_err(|e| CacheSourceFailure::metadata(e, context))?,
        );
        Ok(operation)
    }
}
fn lock_cause<T>(e: TryLockError<T>) -> CacheSourceError {
    match e {
        TryLockError::WouldBlock => CacheSourceError::Busy,
        TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
    }
}
impl PreparedDiskWrite {
    fn descriptors_bytes(&self) -> Option<u64> {
        self.body
            .descriptors
            .iter()
            .try_fold(0u64, |n, d| n.checked_add(u64::try_from(d.nbytes()).ok()?))
    }
}
impl DiskWriteOperation {
    /// A successful immutable output still owns the writer's actual source pin.
    pub(crate) fn completed_source_pin_count(&self) -> usize {
        usize::from(self.committed && self.output.inner.get().is_some())
    }

    /// Borrow the actual file only after its canonical write transition commits.
    /// Completion alone is insufficient if publication rolled back or failed.
    pub(crate) fn committed_file(&self) -> Option<&LiveCacheBlockSource> {
        self.committed.then(|| self.output.result()).flatten()?.ok()
    }

    pub(crate) fn committed_file_control_bytes() -> usize {
        size_of::<(
            &Self,
            Option<Result<&LiveCacheBlockSource, &DiskWriteFailure>>,
            Option<&LiveCacheBlockSource>,
        )>()
    }

    fn failure(&self, cause: Cause) -> DiskWriteOperationFailure {
        DiskWriteOperationFailure {
            cause,
            output: self.output.clone(),
            funding: self.funding.clone(),
        }
    }
    fn validate(&self, state: &CacheManagerState, pending: bool) -> Result<(), Cause> {
        if state.generation != self.key.generation
            || self
                .manager
                .inner
                .disk_worker
                .as_ref()
                .is_none_or(|w| !Arc::ptr_eq(w, &self.worker))
        {
            return Err(Cause::Source(CacheSourceError::Identity));
        }
        let record = state
            .blocks
            .get(&self.key.id)
            .ok_or(Cause::Source(CacheSourceError::Identity))?;
        let phase = if pending {
            CacheStoragePhase::HostWriting
        } else {
            CacheStoragePhase::HostUnbacked
        };
        if record.physical.phase() != phase
            || record.bytes != self.logical_bytes
            || state
                .lifecycle
                .lease_count(&self.key.id)
                .map_err(|e| Cause::Source(e.into()))?
                != self.source_pins
            || state
                .lifecycle
                .source_pin_count(&self.key.id)
                .map_err(|e| Cause::Source(e.into()))?
                != self.source_pins
            || (pending && !record.physical.io_matches(&self.key))
        {
            return Err(Cause::Source(CacheSourceError::Identity));
        }
        let actual = record
            .host_block()
            .ok_or(Cause::Source(CacheSourceError::Identity))?
            .buffers();
        let expected = self
            .host
            .as_ref()
            .ok_or(Cause::Source(CacheSourceError::Identity))?
            .buffers();
        if !std::ptr::eq(actual[0], expected[0]) || !std::ptr::eq(actual[1], expected[1]) {
            return Err(Cause::Source(CacheSourceError::Identity));
        }
        if pending
            && state
                .host_write_reservations
                .get(&self.key)
                .and_then(|r| r.prepared.as_ref())
                .is_none_or(|owner| !Arc::ptr_eq(&owner.inner, &self.occupancy.inner))
        {
            return Err(Cause::Source(CacheSourceError::Identity));
        }
        Ok(())
    }
    /// Same finite queue, exact-key worker admission and canonical begin_write.
    /// No task can execute before the canonical transition has been accepted.
    pub(crate) fn submit(&mut self) -> Result<(), DiskWriteOperationFailure> {
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
            Ok(v) => v,
            Err(e) => {
                let (cause, task) = e.into_parts();
                self.task = Some(task);
                return Err(Cause::Task(cause));
            }
        };
        // A source prepared from stable HostUnbacked cannot be a second owner
        // of an already-pending write. The shared worker still authenticates it.
        if submission.joined {
            self.submission = Some(submission);
            return Err(Cause::Source(CacheSourceError::Identity));
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
                .map_err(|e| Cause::Source(lock_cause(e)))?;
            self.validate(&state, false)?;
            let required = transitions::live_disk_bytes(&state)
                .and_then(|n| n.checked_add(self.logical_bytes))
                .ok_or(Cause::Source(CacheSourceError::Overflow))?;
            if required > state.disk_budget_bytes.unwrap_or(0) {
                return Err(Cause::Policy(CacheResidencyError::BudgetExceeded {
                    tier: CacheTier::Disk,
                    required,
                    budget: state.disk_budget_bytes.unwrap_or(0),
                }));
            }
            if state.host_write_reservations.contains_key(&self.key) {
                return Err(Cause::Source(CacheSourceError::Identity));
            }
            state
                .host_write_reservations
                .validate_prepared_population(
                    state
                        .host_write_reservations
                        .len()
                        .checked_add(1)
                        .ok_or(Cause::Source(CacheSourceError::Overflow))?,
                )
                .map_err(|e| Cause::Source(CacheLifecycleError::from(e).into()))?;
            reporting::update_report_totals_prepared(&mut state).map_err(Cause::Policy)?;
            let ticket = self.ticket.as_ref().expect("prepared task ticket");
            let row = HostWriteReservation {
                reservation_id: NEXT_HOST_WRITE_RESERVATION_ID
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                    .map_err(|_| Cause::Source(CacheSourceError::Overflow))?,
                global_layer: self.key.id.global_layer,
                logical_bytes: self.logical_bytes,
                host_capacity: self.occupancy.host_bytes(),
                ticket: ticket.clone(),
                prepared: Some(self.occupancy.clone()),
            };
            state
                .host_write_reservations
                .insert_prepared(self.key.clone(), row)
                .unwrap_or_else(|_| unreachable!("same locked finite write table"));
            state
                .blocks
                .get_mut(&self.key.id)
                .expect("validated source")
                .physical
                .begin_write(MlxCacheIoOperation {
                    ticket: ticket.clone(),
                    reserved_host_bytes: None,
                    prepared_read: None,
                })
                .expect("same locked unbacked Host source");
            self.armed = true;
            // Independent Disk/transfer reservations were already accepted.
            // This changes telemetry, but not the physical pool population.
            reporting::update_report_totals_prepared(&mut state).map_err(Cause::Policy)?;
        }
        let submission = self.submission.take().expect("prepared exact task");
        self.submitted = true;
        submission.enqueue().map_err(Cause::Worker)?;
        Ok(())
    }
    /// Waits on the exact physical task output, then commits the same shared
    /// finish_write transition. Error keeps the actual file/Host prefix alive.
    pub(crate) fn finish(&mut self) -> Result<(), DiskWriteOperationFailure> {
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
            .expect("submitted ticket")
            .inner
            .with_result(inspect_result)?;
        if !Arc::ptr_eq(&actual.inner, &self.output.inner) {
            return Err(Cause::Completion);
        }
        let completion = self.output.inner.get().ok_or(Cause::Completion)?;
        let location = completion.result.as_ref().map_err(|_| Cause::Write)?;
        let mut state = self
            .manager
            .inner
            .state
            .try_lock()
            .map_err(|e| Cause::Source(lock_cause(e)))?;
        self.validate(&state, true)?;
        let mut occupancy = self
            .occupancy
            .inner
            .reservation
            .try_lock()
            .map_err(|e| Cause::Source(lock_cause(e)))?;
        let reservation = occupancy
            .as_mut()
            .ok_or(Cause::Source(CacheSourceError::Identity))?;
        reporting::update_report_totals_prepared(&mut state).map_err(Cause::Policy)?;
        let overflow = || Cause::Source(CacheSourceError::Overflow);
        let transfer_bytes = state
            .telemetry
            .report
            .transfer_bytes
            .checked_add(self.logical_bytes)
            .ok_or_else(overflow)?;
        let disk_demotions = state
            .telemetry
            .report
            .disk_demotions
            .checked_add(1)
            .ok_or_else(overflow)?;
        let activity = state.layer_activity_mut(self.key.id.global_layer);
        let layer_transfer_bytes = activity
            .transfer_bytes
            .checked_add(self.logical_bytes)
            .ok_or_else(overflow)?;
        let layer_disk_demotions = activity
            .disk_demotions
            .checked_add(1)
            .ok_or_else(overflow)?;
        let (host, io) = state
            .blocks
            .get_mut(&self.key.id)
            .expect("validated pending source")
            .physical
            .finish_write(&self.key, location.clone())
            .expect("same locked pending write");
        let row = state
            .host_write_reservations
            .remove(&self.key)
            .expect("validated pending row");
        if let Err(cause) = reporting::update_report_totals_prepared_replacement(
            &mut state,
            reservation,
            &self.manager.inner.pool_membership,
        ) {
            let mut original = MlxCacheBlockStorage::host(self.key.id.clone(), host, None);
            original
                .begin_write(io)
                .expect("same exact write source rollback");
            let retired = std::mem::replace(
                &mut state
                    .blocks
                    .get_mut(&self.key.id)
                    .expect("same record")
                    .physical,
                original,
            );
            state
                .host_write_reservations
                .insert_prepared(self.key.clone(), row)
                .unwrap_or_else(|_| unreachable!("same vacated finite row"));
            let _ = reporting::update_report_totals_prepared(&mut state);
            drop(state);
            drop(occupancy);
            drop(retired);
            return Err(Cause::Policy(cause));
        }
        state.telemetry.report.transfer_bytes = transfer_bytes;
        state.telemetry.report.disk_demotions = disk_demotions;
        let activity = state.layer_activity_mut(self.key.id.global_layer);
        activity.transfer_bytes = layer_transfer_bytes;
        activity.disk_demotions = layer_disk_demotions;
        self.armed = false;
        self.committed = true;
        drop(state);
        drop(occupancy);
        drop((host, io, row));
        Ok(())
    }
}
fn inspect_result(
    result: Result<&DiskResult, CacheIoBorrowedError<'_>>,
) -> Result<PreparedDiskWriteOutput, Cause> {
    match result {
        Ok(DiskResult::PreparedWrite(output)) if output.result().is_some() => Ok(output.clone()),
        Ok(_) | Err(CacheIoBorrowedError::OperationFailed(_)) => Err(Cause::Completion),
        Err(CacheIoBorrowedError::Execution(cause)) => Err(Cause::Worker(cause.into())),
        Err(CacheIoBorrowedError::Cancelled { generation }) => {
            Err(Cause::Worker(CacheIoWorkerError::Cancelled { generation }))
        }
        Err(CacheIoBorrowedError::Poisoned) => Err(Cause::Source(CacheSourceError::Poisoned)),
    }
}
impl Drop for DiskWriteOperation {
    fn drop(&mut self) {
        if self.armed {
            if let Some(ticket) = &self.ticket {
                ticket.cancel();
            }
            // Teardown may wait for an outstanding borrower, just like source
            // pin teardown. Never leave manager -> ticket -> source-pin cycles.
            let mut state = self
                .manager
                .inner
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let own = state
                .host_write_reservations
                .get(&self.key)
                .and_then(|row| row.prepared.as_ref())
                .is_some_and(|owner| Arc::ptr_eq(&owner.inner, &self.occupancy.inner));
            let removed_io = own
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
            let removed_row = own
                .then(|| state.host_write_reservations.remove(&self.key))
                .flatten();
            // The source's own pin excludes reset/removal throughout this
            // transition. Failed/cancelled writes leave the actual Host source
            // in the same canonical record; no Host charge is refunded here.
            let _ = reporting::update_report_totals_prepared(&mut state);
            drop(state);
            drop((removed_io, removed_row));
        }
        // All native task/result/pin owners are fields and retire after unlock.
    }
}

impl DiskWriteOperation {
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<Cause>(),
            size_of::<DiskWriteOperationFailure>(),
            size_of::<Result<Self, CacheSourceFailure>>(),
            size_of::<Result<(), DiskWriteOperationFailure>>(),
            size_of::<Result<(), Cause>>(),
            size_of::<PreparedCacheIoTask<DiskTask, DiskResult>>(),
            size_of::<RuntimeDiskSubmission>(),
            size_of::<DiskTicket>(),
            size_of::<CacheIoOperationKey>(),
            size_of::<HostWriteReservation>(),
            size_of::<(HostCacheBlock, MlxCacheIoOperation)>(),
            size_of::<MlxCacheBlockStorage>(),
            size_of::<DiskLocation>(),
            size_of::<[u64; 4]>(),
            size_of::<MutexGuard<'_, CacheManagerState>>(),
            size_of::<
                Result<
                    MutexGuard<'_, CacheManagerState>,
                    TryLockError<MutexGuard<'_, CacheManagerState>>,
                >,
            >(),
            size_of::<MutexGuard<'_, Option<CachePoolReservation>>>(),
            size_of::<
                Result<
                    MutexGuard<'_, Option<CachePoolReservation>>,
                    TryLockError<MutexGuard<'_, Option<CachePoolReservation>>>,
                >,
            >(),
            size_of::<(
                bool,
                Option<MlxCacheIoOperation>,
                Option<HostWriteReservation>,
            )>(),
            size_of::<(&InstalledDiskWorker, PreparedDiskWrite, &WorkspaceContext)>(),
            size_of::<(&Self, &CacheManagerState, bool)>(),
            RuntimeCacheIoTicket::<DiskResult>::result_inspection_control_bytes::<
                Result<PreparedDiskWriteOutput, Cause>,
                fn(
                    Result<&DiskResult, CacheIoBorrowedError<'_>>,
                ) -> Result<PreparedDiskWriteOutput, Cause>,
            >()?,
            CacheRecordTable::<CacheIoOperationKey, HostWriteReservation>::mutation_control_bytes(
            )?,
            transitions::live_disk_query_control_bytes()?,
            reporting::report_query_control_bytes()?,
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}

#[cfg(test)]
#[path = "operation/tests.rs"]
mod tests;

#[path = "operation/ordinary_retirement.rs"]
mod ordinary_retirement;

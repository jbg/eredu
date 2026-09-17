//! The existing worker with deferred source jobs and exact host-result handoff.
use super::*;
use crate::backend::runtime::residency::manager::{
    BackgroundHostReadFailure, BackgroundHostReadOwner, PreparedBackgroundHostReads,
    ReadForegroundDiskBatch,
};
use eredu_runtime::working_memory::{HostThreadStartup, HostThreadStartupPlan, WorkingMemoryError};
use eredu_runtime::{BackgroundPrefetchWorkerError, BackgroundThreadFinishError};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
pub(crate) enum BackgroundHostServiceError {
    #[error("{0}")]
    Queue(#[from] BackgroundPrefetchWorkerError<BackgroundHostReadFailure>),
    #[error("{0}")]
    Source(#[from] BackgroundHostReadFailure),
    #[error("{0}")]
    Joined(#[from] BackgroundThreadFinishError<BackgroundHostReadFailure>),
    #[error("background host read was not submitted or has already been consumed")]
    Unscheduled,
}
/// Native manager state remains on the caller. The actual worker only owns
/// the exact detached read and exclusive unsubmitted host writer. Allocation,
/// publication and successful retirement remain on this service's caller.
pub(crate) struct BackgroundHostReadService {
    worker: BackgroundPrefetchWorker<BackgroundHostReadFailure>,
    reads: BackgroundHostReadOwner,
}
fn operation(
    owner: Option<BackgroundHostReadOwner>,
) -> impl Fn(&OffloadUnitId) -> Result<(), BackgroundHostReadFailure> + Send + Sync + 'static {
    move |id| owner.as_ref().expect("runtime host owner").read(id)
}
impl BackgroundHostReadService {
    pub(crate) fn start_control_bytes() -> Option<usize> {
        let operation = operation(None);
        let controls = [
            size_of::<Self>(),
            size_of::<Result<Self, BackgroundHostServiceError>>(),
            size_of::<BackgroundHostServiceError>(),
            size_of::<BackgroundPrefetchReport>(),
            size_of::<Result<BackgroundPrefetchReport, BackgroundHostServiceError>>(),
            BackgroundPrefetchWorker::<BackgroundHostReadFailure>::report_control_bytes()?,
            BackgroundPrefetchWorker::<BackgroundHostReadFailure>::idle_control_bytes()?,
            size_of_val(&operation),
            size_of::<Result<(), BackgroundHostServiceError>>(),
            size_of::<Result<ReadForegroundDiskBatch, BackgroundHostServiceError>>(),
            size_of::<PrefetchDemandResolution<BackgroundHostReadFailure>>(),
            size_of::<(&Self, &OffloadUnitId)>(),
            size_of::<crate::backend::runtime::residency::manager::BackgroundSourceAttempt<'_>>(),
            size_of::<Result<crate::backend::runtime::residency::manager::BackgroundSourceAttempt<'_>, BackgroundHostServiceError>>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
    }
    /// Ordinary thread/PAL and name construction through the same worker.
    /// Original requests use start_admitted with their separate prepaid startup
    /// owner; this ordinary constructor supplies no original thread authority.
    pub(crate) fn start(
        prepared: PreparedBackgroundHostReads,
    ) -> Result<Self, BackgroundHostServiceError> {
        let PreparedBackgroundHostReads { storage, reads } = prepared;
        reads.reserve_controls(Self::start_control_bytes())?;
        let operation = operation(Some(reads.clone()));
        let worker = BackgroundPrefetchWorker::for_prepared_units_retaining(
            storage,
            "eredu-mlx-dense-layer-prefetch",
            operation,
        )?
        .with_nonblocking_drop();
        Ok(Self { worker, reads })
    }
    /// Exact startup plan for the same concrete host-read closure. The caller
    /// accepts this once beside the source/queue controls before invoking start.
    pub(crate) fn thread_plan() -> Result<HostThreadStartupPlan, WorkingMemoryError> {
        BackgroundPrefetchWorker::<BackgroundHostReadFailure>::thread_startup_plan(
            &operation(None),
            "eredu-mlx-dense-layer-prefetch",
        )
    }
    pub(crate) fn start_admitted(
        prepared: PreparedBackgroundHostReads,
        startup: HostThreadStartup,
    ) -> Result<Self, BackgroundHostServiceError> {
        let PreparedBackgroundHostReads { storage, reads } = prepared;
        reads.reserve_controls(Self::start_control_bytes())?;
        let worker = BackgroundPrefetchWorker::for_admitted_prepared_units_retaining(
            storage,
            startup,
            operation(Some(reads.clone())),
        )?
        .with_nonblocking_drop();
        Ok(Self { worker, reads })
    }
    /// Normal retirement observes the real thread join before allowing startup
    /// accounting to retire. Payload mailbox ownership remains independent.
    pub(crate) fn finish(self) -> Result<BackgroundPrefetchReport, BackgroundHostServiceError> {
        let Self { worker, reads } = self;
        // Final source callbacks reach their actual terminal state before the
        // successful report is read. Joining remains the separate startup
        // retirement proof, and all teardown runs before propagating failures.
        let idle = worker.wait_idle();
        let report = worker.report();
        reads.close();
        let joined = worker.finish();
        let retired = reads.discard_ready();
        joined?;
        idle?;
        retired?;
        Ok(report?)
    }
    /// Close before an outer publication/promotion/rollback error escapes.
    /// This never blocks on the worker and never resets source spending.
    pub(crate) fn close(&self) { self.reads.close(); }
    pub(crate) fn attempt(&self) -> Result<crate::backend::runtime::residency::manager::BackgroundSourceAttempt<'_>, BackgroundHostServiceError> {
        Ok(self.reads.attempt()?)
    }
    pub(crate) fn advance_window(&self, active: &[OffloadUnitId]) -> Result<(), BackgroundHostServiceError> {
        let fixed = [size_of::<(&Self, &[OffloadUnitId])>(), size_of::<Result<(), BackgroundHostServiceError>>()];
        self.reads.reserve_controls(fixed.into_iter().try_fold(size_of_val(&fixed), usize::checked_add)
            .and_then(|bytes| bytes.checked_add(BackgroundHostReadOwner::window_control_bytes()?))
            .and_then(|bytes| bytes.checked_add(BackgroundPrefetchWorker::<BackgroundHostReadFailure>::idle_control_bytes()?)))?;
        let attempt = self.attempt()?;
        // Same existing worker/Condvar fence; callback return is sufficient to
        // mutate its mailbox, while startup retirement still requires join.
        self.worker.wait_idle()?;
        self.reads.advance_window(active)?;
        attempt.succeed();
        Ok(())
    }
    pub(crate) fn submit(&self, id: &OffloadUnitId) -> Result<(), BackgroundHostServiceError> {
        let attempt = self.attempt()?;
        self.reads.prepare_submission(id)?;
        self.worker.submit(id, false)?;
        attempt.succeed();
        Ok(())
    }
    pub(crate) fn acquire(
        &self,
        id: &OffloadUnitId,
    ) -> Result<ReadForegroundDiskBatch, BackgroundHostServiceError> {
        let attempt = self.attempt()?;
        let value = match self.worker.wait(id)? {
            PrefetchDemandResolution::Ready => self
                .reads
                .take(id)?
                .ok_or(BackgroundHostServiceError::Unscheduled),
            PrefetchDemandResolution::Failed(cause) => Err(cause.into()),
            PrefetchDemandResolution::Unscheduled => Err(BackgroundHostServiceError::Unscheduled),
        }?;
        attempt.succeed();
        Ok(value)
    }
    pub(crate) fn cancel(&self) -> Result<(), BackgroundHostServiceError> {
        self.close();
        let queue = self.worker.cancel();
        // Actual worker completion has been fenced by cancel even when it
        // returns an operation error. A pre-fence storage/queue refusal leaves
        // the payload mailbox unchanged, just like the queue itself.
        if !matches!(
            &queue,
            Ok(())
                | Err(BackgroundPrefetchWorkerError::OperationFailed { .. })
                | Err(BackgroundPrefetchWorkerError::SourceOperationFailed { .. })
        ) {
            return Err(queue.unwrap_err().into());
        }
        let retired = self.reads.discard_ready();
        queue?;
        Ok(retired?)
    }
    pub(crate) fn report(&self) -> Result<BackgroundPrefetchReport, BackgroundHostServiceError> {
        Ok(self.worker.report()?)
    }
}

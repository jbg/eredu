//! Finite forward generations for the source-only background worker.
use super::{BackgroundHostReadService, BackgroundHostServiceError};
use crate::backend::{
    runtime::residency::manager::{
        BackgroundSourceAttempt, PreparedBackgroundHostReads, PreparedBackgroundHostWindow,
        ResidencyError,
    },
    Error,
};
use eredu_core::residency::BackgroundPrefetchReport;
use eredu_nn::workspace::HostMetadataFunding;
use eredu_runtime::working_memory::{
    HostThreadStartupPlan, OriginalHostSourceCustody, WorkingMemoryReservation,
};
use std::{
    alloc::Layout,
    cell::Cell,
    mem::{size_of, size_of_val},
};

/// All source/read/window destinations are prepared before the forward begins.
/// No thread is started by this value. Each row is consumed exactly once.
pub(crate) struct PreparedBackgroundForward {
    reads: Option<PreparedBackgroundHostReads>,
    windows: Vec<(usize, Option<PreparedBackgroundHostWindow>)>,
    startup: HostThreadStartupPlan,
}
/// Native tensors and the calling thread's buffer budget never enter this
/// owner. Only the exact read service is sent to the existing shared worker.
pub(crate) struct BackgroundHostCoordinator {
    forwards: Vec<Option<PreparedBackgroundForward>>,
    active: Option<(BackgroundHostReadService, PreparedBackgroundForward)>,
    next_forward: usize,
    next_window: usize,
    failed: Cell<bool>,
    failure_issued: Cell<bool>,
    custody: OriginalHostSourceCustody,
    funding: HostMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("background forward order or completion boundary mismatch: {0}")]
    Order(&'static str),
    #[error("background worker: {0}")]
    Service(#[from] BackgroundHostServiceError),
    #[error("background startup: {0}")]
    Memory(#[from] eredu_runtime::working_memory::WorkingMemoryError),
    #[error("background metadata: {0}")]
    Metadata(#[from] eredu_core::HostMetadataFundingError),
}
/// Exact failure and host-only source custody, independent of the bank owner.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct BackgroundCoordinatorFailure {
    #[source]
    cause: Cause,
    custody: OriginalHostSourceCustody,
    funding: HostMetadataFunding,
}
struct Step<'a> {
    failed: &'a Cell<bool>,
    source: Option<BackgroundSourceAttempt<'a>>,
}
impl Step<'_> {
    fn succeed(mut self) {
        self.source.take().expect("active source attempt").succeed();
    }
}
impl Drop for Step<'_> {
    fn drop(&mut self) {
        if self.source.is_some() {
            self.failed.set(true);
        }
    }
}
impl PreparedBackgroundForward {
    /// The supplied Vec is the actual paid final window destination, moved
    /// unchanged. Its caller prices each window/read producer independently.
    pub(crate) fn from_prepared(
        reads: PreparedBackgroundHostReads,
        windows: Vec<(usize, Option<PreparedBackgroundHostWindow>)>,
        startup: HostThreadStartupPlan,
    ) -> Self {
        Self {
            reads: Some(reads),
            windows,
            startup,
        }
    }
}
impl BackgroundHostCoordinator {
    pub(crate) fn is_idle(&self) -> bool {
        !self.failed.get() && self.active.is_none()
    }
    /// Final coordinator and failure/transition frames only. Source, queue,
    /// windows, Vec backing and the separate thread startup have their own
    /// producer queries and are not counted twice in this amount.
    pub(crate) fn control_bytes() -> Option<usize> {
        let fixed = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<PreparedBackgroundForward>(),
            size_of::<Option<PreparedBackgroundForward>>(),
            size_of::<BackgroundCoordinatorFailure>(),
            size_of::<Box<BackgroundCoordinatorFailure>>(),
            Layout::new::<BackgroundCoordinatorFailure>().size(),
            size_of::<Cause>(),
            size_of::<Step<'_>>(),
            size_of::<Result<(), Error>>(),
            size_of::<BackgroundPrefetchReport>(),
            size_of::<Result<BackgroundPrefetchReport, Error>>(),
            size_of::<Result<Option<BackgroundPrefetchReport>, Error>>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Result<BackgroundHostReadService, BackgroundHostServiceError>>(),
            size_of::<
                Result<
                    eredu_runtime::working_memory::HostThreadStartup,
                    eredu_runtime::working_memory::WorkingMemoryError,
                >,
            >(),
            size_of::<Result<BackgroundSourceAttempt<'_>, BackgroundHostServiceError>>(),
            size_of::<Option<(BackgroundHostReadService, PreparedBackgroundForward)>>(),
            size_of::<(&mut Self, Option<&WorkingMemoryReservation>)>(),
            size_of::<std::slice::Iter<'_, Option<PreparedBackgroundForward>>>(),
            size_of::<std::slice::Iter<'_, (usize, Option<PreparedBackgroundHostWindow>)>>(),
        ];
        fixed
            .into_iter()
            .try_fold(size_of_val(&fixed), usize::checked_add)
    }
    pub(crate) fn from_prepared(
        forwards: Vec<Option<PreparedBackgroundForward>>,
        custody: OriginalHostSourceCustody,
        funding: HostMetadataFunding,
    ) -> Result<Self, Error> {
        // A funding refusal returns the fixed native planning error. It must
        // not allocate an unaccepted failure Box while reporting that refusal.
        let bytes = Self::control_bytes().ok_or(Error::WorkspacePlanning(
            eredu_core::HostMetadataFundingError::Overflow,
        ))?;
        funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        let value = Self {
            forwards,
            active: None,
            next_forward: 0,
            next_window: 0,
            failed: Cell::new(false),
            failure_issued: Cell::new(false),
            custody,
            funding,
        };
        if value.forwards.iter().any(|row| {
            row.as_ref().is_none_or(|forward| {
                forward.reads.is_none()
                    || forward.windows.iter().any(|(_, window)| window.is_none())
            })
        }) {
            return Err(value.fail(Cause::Order("incomplete prepared destinations")));
        }
        Ok(value)
    }
    fn error(
        cause: Cause,
        issued: &Cell<bool>,
        custody: &OriginalHostSourceCustody,
        funding: &HostMetadataFunding,
    ) -> Error {
        // Exactly one paid failure destination. Retrying a terminal coordinator
        // cannot allocate another Box while an earlier error remains alive.
        if issued.replace(true) {
            return Error::PrefillScopeUnavailable;
        }
        Error::Residency(ResidencyError::OriginalBackgroundCoordinator(Box::new(
            BackgroundCoordinatorFailure {
                cause,
                custody: custody.clone(),
                funding: funding.clone(),
            },
        )))
    }
    fn fail(&self, cause: Cause) -> Error {
        self.close();
        Self::error(cause, &self.failure_issued, &self.custody, &self.funding)
    }
    /// No lock, cancellation wait or source reset on abort/unwind. Existing
    /// worker Drop retains source/queue/startup until its actual retirement.
    pub(crate) fn close(&self) {
        self.failed.set(true);
        if let Some((service, _)) = &self.active {
            service.close();
        }
    }
    fn begin(&mut self, reservation: Option<&WorkingMemoryReservation>) -> Result<(), Error> {
        if self.failed.get() || self.active.is_some() {
            return Err(self.fail(Cause::Order("begin while active or failed")));
        }
        self.custody
            .validate_account(reservation)
            .map_err(|cause| self.fail(cause.into()))?;
        let mut forward = match self
            .forwards
            .get_mut(self.next_forward)
            .and_then(Option::take)
        {
            Some(value) => value,
            None => return Err(self.fail(Cause::Order("prepared forward exhausted"))),
        };
        let startup = forward
            .startup
            .prepare_for_source(&self.custody, reservation)
            .map_err(|cause| self.fail(cause.into()))?;
        let reads = forward
            .reads
            .take()
            .ok_or_else(|| self.fail(Cause::Order("forward read source already consumed")))?;
        let service = BackgroundHostReadService::start_admitted(reads, startup)
            .map_err(|cause| self.fail(cause.into()))?;
        self.next_window = 0;
        self.active = Some((service, forward));
        Ok(())
    }
    /// Execute the same caller's acquisition/population closure. It must include
    /// every error point before the window is accepted. The source and request
    /// failure guards remain active until that exact closure returns success.
    pub(crate) fn with_window<T, F>(
        &mut self,
        index: usize,
        reservation: Option<&WorkingMemoryReservation>,
        execute: F,
    ) -> Result<T, Error>
    where
        F: FnOnce(&BackgroundHostReadService, PreparedBackgroundHostWindow) -> Result<T, Error>,
    {
        let controls = [
            size_of::<F>(),
            size_of::<T>(),
            size_of::<Result<T, Error>>(),
            size_of::<Option<usize>>(),
            size_of::<(&mut Self, usize, Option<&WorkingMemoryReservation>, F)>(),
        ];
        if self.failed.get() {
            return Err(Error::PrefillScopeUnavailable);
        }
        let bytes = match controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
        {
            Some(value) => value,
            None => {
                self.close();
                return Err(Error::WorkspacePlanning(
                    eredu_core::HostMetadataFundingError::Overflow,
                ));
            }
        };
        if let Err(cause) = self.funding.reserve_metadata(bytes) {
            self.close();
            return Err(Error::WorkspacePlanning(cause));
        }
        if self.active.is_none() {
            let first = self
                .forwards
                .get(self.next_forward)
                .and_then(Option::as_ref)
                .and_then(|forward| forward.windows.first())
                .map(|(ordinal, _)| *ordinal);
            if first != Some(index) {
                return Err(self.fail(Cause::Order("unexpected first window ordinal")));
            }
            self.begin(reservation)?;
        }
        let next = self
            .next_window
            .checked_add(1)
            .ok_or_else(|| self.fail(Cause::Order("window ordinal overflow")))?;
        let (service, forward) = self.active.as_mut().expect("begun forward");
        let fail = |cause| {
            self.failed.set(true);
            service.close();
            Self::error(cause, &self.failure_issued, &self.custody, &self.funding)
        };
        let window = match forward.windows.get_mut(self.next_window) {
            Some((ordinal, window)) if *ordinal == index => match window.take() {
                Some(value) => value,
                None => return Err(fail(Cause::Order("window source already consumed"))),
            },
            _ => return Err(fail(Cause::Order("unexpected next window ordinal"))),
        };
        let source = match service.attempt() {
            Ok(value) => value,
            Err(cause) => return Err(fail(cause.into())),
        };
        let step = Step {
            failed: &self.failed,
            source: Some(source),
        };
        let value = execute(service, window)?;
        self.next_window = next;
        step.succeed();
        Ok(value)
    }
    /// Called only after the operation bank checked its real pending-lease queue
    /// is empty. No queue-only cancellation state permits a new generation.
    pub(crate) fn finish_forward(&mut self) -> Result<BackgroundPrefetchReport, Error> {
        if self.failed.get() {
            return Err(self.fail(Cause::Order("finish after failure")));
        }
        // A source-qualified empty visit has no thread or read to complete.
        // Its prepared, unused destinations retire without starting a worker.
        if self.active.is_none()
            && self
                .forwards
                .get(self.next_forward)
                .and_then(Option::as_ref)
                .is_some_and(|forward| forward.windows.is_empty())
        {
            let forward = self.forwards[self.next_forward]
                .take()
                .expect("selected empty forward");
            self.next_forward = self
                .next_forward
                .checked_add(1)
                .ok_or_else(|| self.fail(Cause::Order("forward ordinal overflow")))?;
            drop(forward);
            return Ok(BackgroundPrefetchReport::default());
        }
        let complete = self.active.as_ref().is_some_and(|(_, forward)| {
            self.next_window == forward.windows.len()
                && forward.windows.iter().all(|(_, window)| window.is_none())
        });
        if !complete {
            return Err(self.fail(Cause::Order("finish before all selected windows")));
        }
        let (service, forward) = self.active.take().expect("complete forward");
        // Actual thread join precedes advancing the request generation. A read
        // error, incomplete window or failed join leaves the request terminal.
        let report = service.finish().map_err(|cause| self.fail(cause.into()))?;
        drop(forward);
        self.next_forward = self
            .next_forward
            .checked_add(1)
            .ok_or_else(|| self.fail(Cause::Order("forward ordinal overflow")))?;
        self.next_window = 0;
        Ok(report)
    }
}
impl Drop for BackgroundHostCoordinator {
    fn drop(&mut self) {
        self.close();
    }
}

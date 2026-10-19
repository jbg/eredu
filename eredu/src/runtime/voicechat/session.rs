//! Controlled and iterator-driven advancement of one ordinary scheduler request.

use std::{
    cell::RefCell,
    error::Error,
    fmt,
    rc::Rc,
    sync::Arc,
    time::{Duration, Instant},
};

use eredu_core::{
    scheduler::{
        RequestId, RequestStatus, Scheduler, SchedulerError, SchedulerLimits, SchedulerProgress,
        SchedulerReport, SemanticStateTransaction,
    },
    BackendFailure,
};

use super::{
    VoiceChatAdvance, VoiceChatBlocked, VoiceChatExecutionError, VoiceChatPreparation,
    VoiceChatPublicationBranch, VoiceChatPublicationCapacity, VoiceChatPublicationError,
    VoiceChatPublicationState, VoiceChatReminderPlayback, VoiceChatSubmittedWork,
    VoiceChatToolTicket, VoiceChatToolWaitMessages, VoiceChatToolWaitPolicy, VoiceChatWork,
    VoiceChatWorkKind,
};

mod fork;
mod retained_failure;
mod snapshot;
pub use super::publication::snapshot::VoiceChatSnapshotError;
pub use fork::VoiceChatFork;
use retained_failure::{capture, FailureSlot, TrackedOutput, TrackedState};
pub use snapshot::VoiceChatSessionSnapshot;

/// Bounded session composition over already prepared neural and conversation state.
#[derive(Debug, Clone, Copy)]
pub struct VoiceChatSessionConfig {
    /// Caller-selected origin mapping recorded durations to scheduler timestamps.
    pub clock_origin: Instant,
    /// Explicit time attributed to prompt/Aria prefill.
    pub start_at: Duration,
    /// Framed backbone positions from `VoiceChatExecutionInputs`.
    pub backbone_prompt_positions: u64,
    /// Reserved payload bound for each invocation, within the owner's queue limits.
    pub publication_capacity: VoiceChatPublicationCapacity,
}

/// Current session lifecycle. Retirement describes scheduler-owned submitted work,
/// not independently retained backend recovery resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceChatSessionStatus {
    /// No invocation is in flight; explicit advancement may prepare the next one.
    Ready,
    /// Exact native completion is pending; PCM ingress remains available.
    Pending,
    /// No work is ready under the recorded input/time/tool policy.
    Blocked(VoiceChatBlocked),
    /// Terminal codec drain committed and canonical state was released.
    Finished,
    /// Cancellation is terminal, with possible submitted work awaiting retirement.
    Cancelled {
        /// Whether the scheduler still owns submitted work.
        retiring: bool,
    },
    /// The request failed; the backend retains any independent recovery authority.
    Failed {
        /// Whether the scheduler still owns submitted work.
        retiring: bool,
    },
}

impl VoiceChatSessionStatus {
    pub(super) fn settled(self) -> bool {
        matches!(
            self,
            Self::Finished | Self::Cancelled { retiring: false } | Self::Failed { retiring: false }
        )
    }
}

/// One advancement result. Committed events remain in the independent receiver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoiceChatSessionProgress {
    /// Newly submitted invocations; at most one per advancement.
    pub submitted: usize,
    /// Newly committed invocations; polling a pending invocation submits no successor.
    pub committed: usize,
    /// Readiness or terminal state after this advancement.
    pub status: VoiceChatSessionStatus,
}

/// Scheduler context plus the original portable failure, if a mechanism failed.
#[derive(Debug)]
pub struct VoiceChatSessionFailure {
    context: SchedulerError,
    original: Option<Arc<BackendFailure>>,
}
impl VoiceChatSessionFailure {
    /// Structured scheduler context, including cleanup diagnostics.
    pub fn context(&self) -> &SchedulerError {
        &self.context
    }
    /// Original typed native or publication failure retained before formatting.
    pub fn original(&self) -> Option<&BackendFailure> {
        self.original.as_deref()
    }
}
impl fmt::Display for VoiceChatSessionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.context.fmt(f)
    }
}
impl Error for VoiceChatSessionFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.original
            .as_deref()
            .map(|error| error as &(dyn Error + 'static))
    }
}

/// Backend-independent controlled-session rejection.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatSessionError {
    /// Recorded time is earlier than a previous session operation.
    #[error("VoiceChat session time moved backwards")]
    Time,
    /// Saved state could not be captured or restored.
    #[error(transparent)]
    Snapshot(#[from] VoiceChatSnapshotError),
    /// The recorded timestamp cannot be represented relative to the supplied origin.
    #[error("VoiceChat session timestamp exceeds clock capacity")]
    ClockCapacity,
    /// A queued or submitted branch owns the conversation admission.
    #[error("VoiceChat session has outstanding work")]
    Busy,
    /// Canonical state is no longer available for host-policy mutation.
    #[error("VoiceChat session is terminal ({0:?})")]
    Terminal(RequestStatus),
    /// Recoverable preparation, queue-capacity or host-policy rejection.
    #[error(transparent)]
    Publication(#[from] VoiceChatPublicationError),
    /// The scheduler has rejected or failed an operation, with original sources retained.
    #[error(transparent)]
    Scheduler(#[from] VoiceChatSessionFailure),
}

/// One controlled session using the shared scheduler and prepared-work driver.
///
/// Construct the publication owner with external PCM ingress and keep its sender
/// and receiver independently. They accept input and deliver committed events
/// while this owner retains pending work, and survive request cancellation. This
/// composition performs no model loading, tool execution or hidden clock reads.
pub struct VoiceChatSession<S, D>
where
    S: SemanticStateTransaction,
    S::Error: Send + Sync + 'static,
{
    scheduler: Scheduler<VoiceChatWork<D>, TrackedState<S, D>, TrackedOutput>,
    request: RequestId,
    config: VoiceChatSessionConfig,
    last_time: Duration,
    host_actions: Rc<()>,
    snapshot_budget: Option<eredu_runtime::execution_control::SnapshotBudget>,
    failure: FailureSlot,
    lineage: Rc<RefCell<fork::ForkLineage>>,
}

impl<S, D> VoiceChatSession<S, D>
where
    S: SemanticStateTransaction,
    S::Error: Send + Sync + 'static,
{
    /// Registers one prepared owner and queues explicit prompt/Aria prefill.
    /// Neural execution starts only on `advance`; errors submit no native work.
    pub fn start(
        mut state: VoiceChatPublicationState<S, D>,
        config: VoiceChatSessionConfig,
    ) -> Result<Self, VoiceChatSessionError> {
        config
            .clock_origin
            .checked_add(config.start_at)
            .ok_or(VoiceChatSessionError::ClockCapacity)?;
        state.require_input()?;
        let request = state.conversation().turn_identity().request();
        let work = state.prepare_prefill(
            config.backbone_prompt_positions,
            config.start_at,
            config.publication_capacity,
        )?;
        let mut session = Self::register_idle(
            state,
            config,
            config.start_at,
            None,
            Rc::new(RefCell::new(fork::ForkLineage::new(request))),
        )?;
        session
            .scheduler
            .enqueue(request, work)
            .map_err(|error| session.scheduler_error(error))?;
        Ok(session)
    }

    fn register_idle(
        state: VoiceChatPublicationState<S, D>,
        config: VoiceChatSessionConfig,
        last_time: Duration,
        snapshot_budget: Option<eredu_runtime::execution_control::SnapshotBudget>,
        lineage: Rc<RefCell<fork::ForkLineage>>,
    ) -> Result<Self, VoiceChatSessionError> {
        let request = state.conversation().turn_identity().request();
        let failure = FailureSlot::default();
        let build_error = |context| VoiceChatSessionFailure {
            context,
            original: None,
        };
        let mut scheduler = Scheduler::new(SchedulerLimits::new(1, 1).map_err(build_error)?)
            .map_err(build_error)?;
        scheduler
            .register(
                request,
                TrackedState {
                    inner: state,
                    failure: failure.clone(),
                },
            )
            .map_err(build_error)?;
        Ok(Self {
            scheduler,
            request,
            config,
            last_time,
            host_actions: Rc::new(()),
            snapshot_budget,
            failure,
            lineage,
        })
    }

    fn timestamp(&mut self, at: Duration) -> Result<Instant, VoiceChatSessionError> {
        if at < self.last_time {
            return Err(VoiceChatSessionError::Time);
        }
        let now = self
            .config
            .clock_origin
            .checked_add(at)
            .ok_or(VoiceChatSessionError::ClockCapacity)?;
        self.last_time = at;
        Ok(now)
    }

    fn scheduler_error(&self, context: SchedulerError) -> VoiceChatSessionError {
        VoiceChatSessionFailure {
            context,
            original: self.failure.borrow_mut().take(),
        }
        .into()
    }

    /// Reports the scheduler's current lifecycle without polling native work.
    pub fn status(&self) -> VoiceChatSessionStatus {
        let retiring = self.scheduler.report().current_in_flight_work != 0;
        match self.scheduler.request_status(self.request) {
            Some(RequestStatus::Finished) => VoiceChatSessionStatus::Finished,
            Some(RequestStatus::Cancelled) => VoiceChatSessionStatus::Cancelled { retiring },
            Some(RequestStatus::Failed | RequestStatus::DeadlineExceeded) | None => {
                VoiceChatSessionStatus::Failed { retiring }
            }
            Some(RequestStatus::Active) if retiring => VoiceChatSessionStatus::Pending,
            Some(RequestStatus::Active) => VoiceChatSessionStatus::Ready,
        }
    }

    /// Current bounded scheduler telemetry, including cancellation retirement.
    pub fn report(&self) -> SchedulerReport {
        self.scheduler.report()
    }

    /// Whether the committed conversation is serializing a tool call, waiting
    /// for its result or decoding the response through its end marker.
    /// Pending invocations do not expose tentative changes. Returns false after
    /// the scheduler releases the canonical request state.
    pub fn has_tool_work(&self) -> bool {
        self.scheduler
            .request_state(self.request)
            .is_some_and(|state| state.inner.conversation().has_tool_work())
    }

    fn observe(
        &mut self,
        progress: SchedulerProgress<VoiceChatWork<D>, TrackedOutput>,
    ) -> Result<VoiceChatSessionProgress, VoiceChatSessionError> {
        let committed = progress.committed.len();
        let drained = progress
            .committed
            .iter()
            .any(|(_, work, _)| work.kind() == VoiceChatWorkKind::Drain);
        if let Some((_, error)) = progress.failed.into_iter().next() {
            return Err(self.scheduler_error(error));
        }
        // Host output is already retained by the receiver. Drop committed native
        // gates before selecting another frame from the same execution resource.
        drop(progress.committed);
        if drained {
            self.scheduler
                .finish(self.request)
                .map_err(|error| self.scheduler_error(error))?;
        }
        Ok(VoiceChatSessionProgress {
            submitted: progress.newly_submitted,
            committed,
            status: self.status(),
        })
    }

    /// Polls exact completion without selecting or submitting another invocation.
    /// Continue polling cancelled requests until scheduler-owned work retires.
    pub fn poll(
        &mut self,
        at: Duration,
    ) -> Result<VoiceChatSessionProgress, VoiceChatSessionError> {
        let now = self.timestamp(at)?;
        let progress = self.scheduler.poll_completions(now);
        self.observe(progress)
    }

    /// Advances at most one invocation through the shared scheduler. The execution
    /// closure calls `work.submit` or the MLX binding; it owns no host policy.
    pub fn advance(
        &mut self,
        advance: VoiceChatAdvance,
        mut execute: impl FnMut(
            &VoiceChatWork<D>,
            &mut VoiceChatPublicationBranch<S::Branch, D>,
        ) -> Result<VoiceChatSubmittedWork, VoiceChatExecutionError>,
    ) -> Result<VoiceChatSessionProgress, VoiceChatSessionError> {
        let now = self.timestamp(advance.at)?;
        let polled = self.scheduler.poll_completions(now);
        let observed = self.observe(polled)?;
        if observed.committed != 0 || observed.status != VoiceChatSessionStatus::Ready {
            return Ok(observed);
        }
        if self.scheduler.queued_for_request(self.request) == 0 {
            let state = match self.scheduler.request_state_mut(self.request) {
                Ok(state) => state,
                Err(context) => return Err(self.scheduler_error(context)),
            };
            let prepared = state
                .inner
                .prepare_next(advance, self.config.publication_capacity)?;
            match prepared {
                VoiceChatPreparation::Ready(work) => {
                    self.scheduler
                        .enqueue(self.request, work)
                        .map_err(|error| self.scheduler_error(error))?;
                }
                VoiceChatPreparation::Blocked(reason) => {
                    return Ok(VoiceChatSessionProgress {
                        submitted: 0,
                        committed: 0,
                        status: VoiceChatSessionStatus::Blocked(reason),
                    });
                }
            }
        }
        self.scheduler
            .prepare_bounded(1, now)
            .map_err(|error| self.scheduler_error(error))?;
        let failure = self.failure.clone();
        let submitted = self
            .scheduler
            .submit_prepared(now, |_, work, branch| {
                execute(work, &mut branch.inner)
                    .map(|inner| TrackedOutput {
                        inner,
                        failure: failure.clone(),
                    })
                    .map_err(|error| capture(&failure, error))
            })
            .map_err(|error| self.scheduler_error(error))?;
        let mut progress = self.scheduler.poll_completions(now);
        progress.newly_submitted = submitted;
        self.observe(progress)
    }

    fn mutable_state(
        &mut self,
    ) -> Result<&mut VoiceChatPublicationState<S, D>, VoiceChatSessionError> {
        if let Some(status) = self.scheduler.request_status(self.request) {
            if status != RequestStatus::Active {
                return Err(VoiceChatSessionError::Terminal(status));
            }
        }
        let report = self.scheduler.report();
        if report.queued_work != 0
            || report.prepared_work != 0
            || report.current_in_flight_work != 0
        {
            return Err(VoiceChatSessionError::Busy);
        }
        match self.scheduler.request_state_mut(self.request) {
            Ok(state) => Ok(&mut state.inner),
            Err(context) => Err(VoiceChatSessionFailure {
                context,
                original: self.failure.borrow_mut().take(),
            }
            .into()),
        }
    }

    /// Submits a result only for its committed tool ticket, between invocations.
    pub fn submit_response(
        &mut self,
        ticket: VoiceChatToolTicket,
        text: &str,
    ) -> Result<(), VoiceChatSessionError> {
        self.mutable_state()?.submit_response(ticket, text)?;
        self.host_actions = Rc::new(());
        Ok(())
    }

    /// Begins recorded tool waiting after a complete call batch commits.
    pub fn start_wait(
        &mut self,
        ticket: VoiceChatToolTicket,
        at: Duration,
        policy: VoiceChatToolWaitPolicy,
        messages: VoiceChatToolWaitMessages<'_>,
    ) -> Result<(), VoiceChatSessionError> {
        self.timestamp(at)?;
        self.mutable_state()?
            .start_wait(ticket, at, policy, messages)?;
        self.host_actions = Rc::new(());
        Ok(())
    }

    /// Records actual reminder presentation completion.
    pub fn finish_playback(
        &mut self,
        receipt: VoiceChatReminderPlayback,
        at: Duration,
    ) -> Result<(), VoiceChatSessionError> {
        self.timestamp(at)?;
        self.mutable_state()?.finish_playback(receipt, at)?;
        self.host_actions = Rc::new(());
        Ok(())
    }

    /// Invalidates tool/turn work after any outstanding invocation retires.
    pub fn interrupt(&mut self) -> Result<(), VoiceChatSessionError> {
        self.mutable_state()?.interrupt()?;
        self.host_actions = Rc::new(());
        Ok(())
    }

    /// Cancels using recorded time, retaining submitted resources until retirement.
    /// Independently owned committed output remains deliverable.
    pub fn cancel(
        &mut self,
        at: Duration,
    ) -> Result<VoiceChatSessionStatus, VoiceChatSessionError> {
        let now = self.timestamp(at)?;
        if self.scheduler.request_status(self.request) == Some(RequestStatus::Active) {
            self.scheduler
                .cancel_at(self.request, now)
                .map_err(|error| self.scheduler_error(error))?;
        }
        Ok(self.status())
    }

    /// Repeatedly advances this same session with caller-recorded ticks. The
    /// iterator neither sleeps nor reads a clock; pending work remains owned if
    /// ticks run out. The independent receiver can be drained between yields.
    pub fn drive<'a, I, F>(
        &'a mut self,
        ticks: I,
        mut execute: F,
    ) -> impl Iterator<Item = Result<VoiceChatSessionProgress, VoiceChatSessionError>> + 'a
    where
        I: IntoIterator<Item = VoiceChatAdvance>,
        I::IntoIter: 'a,
        F: FnMut(
                &VoiceChatWork<D>,
                &mut VoiceChatPublicationBranch<S::Branch, D>,
            ) -> Result<VoiceChatSubmittedWork, VoiceChatExecutionError>
            + 'a,
    {
        let mut ticks = ticks.into_iter();
        std::iter::from_fn(move || {
            if self.status().settled() {
                return None;
            }
            Some(self.advance(ticks.next()?, &mut execute))
        })
    }
}

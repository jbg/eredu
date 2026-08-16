//! Architecture-neutral, event-tracked request scheduling.
//!
//! A transition executes against a semantic branch.  Its branch, output,
//! retained resources, and exact backend completion remain transition-owned
//! until completion.  Canonical request state changes only when that exact
//! completion succeeds.  Cancellation can therefore discard queued or
//! prepared work, or abandon submitted work without publishing it.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    time::{Duration, Instant},
};

use safemlx::{
    distributed::{self, Group},
    transforms::async_eval_with_event,
    Array, EventBackend, Stream,
};

use crate::error::Error;
pub use safemlx_lm_core::scheduler::{
    CancellationCause, RequestId, RequestStatus, SchedulerLimits, WorkId, WorkLifecycle,
};

/// Exact cross-rank descriptor for a program-specific work payload.
pub trait WorkDescriptor {
    /// Appends every semantic field in stable wire order.
    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Error>;

    /// Number of program-defined non-preemptible operations in this transition.
    fn execution_slice_size(&self) -> usize {
        1
    }
}

/// A branch/delta transaction over canonical request state.
///
/// Implementations should share immutable array backing and copy only mutable
/// semantic metadata.  `commit_branch` is the sole publication boundary.
pub trait SemanticStateTransaction {
    /// Transition-local state or delta.
    type Branch;

    /// Creates a branch without mutating canonical state.
    fn branch(&self) -> Result<Self::Branch, Error>;

    /// Atomically publishes one successfully completed branch.
    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Error>;

    /// Discards a completed or failed branch and rolls back shared deltas.
    fn discard_branch(branch: Self::Branch) -> Result<(), Error> {
        drop(branch);
        Ok(())
    }

    /// Whether another branch may be based on the current canonical state.
    ///
    /// Autoregressive implementations should keep the default. Programs with
    /// independent mergeable deltas may explicitly permit more concurrency.
    fn permits_parallel_branches(&self) -> bool {
        false
    }
}

/// Exact completion and resource ownership for one submitted transition.
///
/// This is an observation contract over the existing SafeMLX event, not a new
/// event type. Implementations must retain every array and lease needed by the
/// submitted operation until this value is dropped.
pub trait TransitionOutput {
    /// Nonblocking exact-completion query.
    fn is_complete(&self) -> Result<bool, Error>;

    /// Backend which owns the exact completion.
    fn backend(&self) -> Result<EventBackend, Error>;

    /// Whether already executing work can be physically interrupted.
    fn physically_preemptible(&self) -> bool {
        false
    }

    /// Number of explicitly retained arrays and transfer/resource leases.
    fn retained_resources(&self) -> usize;
}

/// Static scheduler capabilities and configured bounds.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SchedulerCapabilities {
    /// Configured scheduler limits.
    pub limits: SchedulerLimits,
    /// Backends observed on submitted transitions.
    pub observed_backends: Vec<EventBackend>,
    /// Whether every observed backend output reports physical preemption.
    pub executing_work_physically_preemptible: bool,
    /// Exact description of the unavoidable cancellation interval.
    pub non_preemptible_interval: &'static str,
}

/// Snapshot of lifecycle, occupancy, cancellation, and latency telemetry.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SchedulerReport {
    /// Requests that currently own canonical state.
    pub active_requests: usize,
    /// Work not yet prepared.
    pub queued_work: usize,
    /// Work prepared but not submitted.
    pub prepared_work: usize,
    /// Non-abandoned submitted work awaiting exact completion.
    pub submitted_in_flight_work: usize,
    /// Completed work awaiting publication.
    pub completing_work: usize,
    /// Abandoned submitted work retaining resources.
    pub abandoned_in_flight_work: usize,
    /// Failed distributed work retaining endpoints until every rank is safe.
    pub failed_in_flight_work: usize,
    /// Current submitted work, including abandoned work.
    pub current_in_flight_work: usize,
    /// Peak submitted work.
    pub peak_in_flight_work: usize,
    /// Peak accepted nonterminal work.
    pub peak_queued_work: usize,
    /// Total accepted work.
    pub submitted_work: u64,
    /// Total committed work.
    pub completed_work: u64,
    /// Total failed work.
    pub failed_work: u64,
    /// Work discarded before submission.
    pub discarded_work: u64,
    /// Work cancelled before backend submission.
    pub cancellation_before_submission: u64,
    /// Work cancelled after backend submission.
    pub cancellation_after_submission: u64,
    /// Total abandoned work released after exact completion.
    pub abandoned_released_work: u64,
    /// Resources currently retained by abandoned work.
    pub abandoned_retained_resources: usize,
    /// Maximum retained abandoned resources.
    pub peak_abandoned_retained_resources: usize,
    /// Last observed cancellation-to-release latency in nanoseconds.
    pub last_cancellation_to_release_ns: Option<u128>,
    /// Maximum observed cancellation-to-release latency in nanoseconds.
    pub max_cancellation_to_release_ns: Option<u128>,
    /// Requests ended normally.
    pub finished_requests: u64,
    /// Requests cancelled explicitly.
    pub cancelled_requests: u64,
    /// Requests ended by deadline.
    pub deadline_expired_requests: u64,
    /// Scheduler turns which submitted at least one transition.
    pub drain_cycles: u64,
    /// Maximum new submissions allowed per turn.
    pub configured_submission_bound: usize,
    /// Program-defined execution slice bound.
    pub configured_slice_bound: usize,
    /// Whether unsafe distributed ordering poisoned the scheduler.
    pub poisoned: bool,
}

/// One successfully committed scheduler transition.
#[derive(Debug)]
pub struct CompletedWork<W, O> {
    id: WorkId,
    work: W,
    output: O,
}

impl<W, O> CompletedWork<W, O> {
    /// Returns the scheduler-assigned work identity.
    pub const fn id(&self) -> WorkId {
        self.id
    }

    /// Returns the program-specific work payload.
    pub const fn work(&self) -> &W {
        &self.work
    }

    /// Returns the completed output.
    pub const fn output(&self) -> &O {
        &self.output
    }

    /// Consumes the completion into identity, work, and output.
    pub fn into_parts(self) -> (WorkId, W, O) {
        (self.id, self.work, self.output)
    }
}

/// Request-local asynchronous failure observed during a scheduler turn.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FailedWork {
    /// Failed work identity.
    pub id: WorkId,
    /// Stable error text retained for telemetry and caller reporting.
    pub error: String,
}

/// Outputs and failures resolved by one scheduler turn.
#[derive(Debug)]
pub struct SchedulerProgress<W, O> {
    /// Successfully committed outputs.
    pub committed: Vec<CompletedWork<W, O>>,
    /// Request-local failures; unrelated requests remain schedulable.
    pub failed: Vec<FailedWork>,
    /// Number of new transitions submitted by this turn.
    pub newly_submitted: usize,
}

impl<W, O> Default for SchedulerProgress<W, O> {
    fn default() -> Self {
        Self {
            committed: Vec::new(),
            failed: Vec::new(),
            newly_submitted: 0,
        }
    }
}

#[derive(Debug)]
struct RequestEntry<W, S> {
    state: S,
    pending: VecDeque<QueuedWork<W>>,
    next_sequence: u64,
}

#[derive(Debug)]
struct QueuedWork<W> {
    id: WorkId,
    work: W,
    deadline: Option<Instant>,
}

#[derive(Debug)]
struct PreparedWork<W, B> {
    id: WorkId,
    work: W,
    descriptor: Vec<u32>,
    branch: B,
    deadline: Option<Instant>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum SubmittedDisposition {
    Publish,
    Abandon {
        cause: CancellationCause,
        cancelled_at: Instant,
    },
    Fail,
}

#[derive(Debug)]
struct SubmittedWork<W, B, O> {
    id: WorkId,
    work: W,
    branch: B,
    output: O,
    disposition: SubmittedDisposition,
}

/// Bounded fair scheduler with exact-completion publication.
#[derive(Debug)]
pub struct FairScheduler<W, S, O>
where
    S: SemanticStateTransaction,
    O: TransitionOutput,
{
    limits: SchedulerLimits,
    requests: BTreeMap<RequestId, RequestEntry<W, S>>,
    terminal: BTreeMap<RequestId, RequestStatus>,
    ready: VecDeque<RequestId>,
    prepared: VecDeque<PreparedWork<W, S::Branch>>,
    in_flight: Vec<SubmittedWork<W, S::Branch, O>>,
    lifecycle: BTreeMap<WorkId, WorkLifecycle>,
    accepted_work: usize,
    peak_accepted_work: usize,
    peak_in_flight_work: usize,
    submitted_work: u64,
    completed_work: u64,
    failed_work: u64,
    discarded_work: u64,
    cancellation_before_submission: u64,
    cancellation_after_submission: u64,
    abandoned_released_work: u64,
    peak_abandoned_retained_resources: usize,
    last_cancellation_to_release: Option<Duration>,
    max_cancellation_to_release: Option<Duration>,
    finished_requests: u64,
    cancelled_requests: u64,
    deadline_expired_requests: u64,
    drain_cycles: u64,
    observed_backends: BTreeSet<u8>,
    all_outputs_preemptible: bool,
    poisoned: Option<String>,
}

impl<W, S, O> FairScheduler<W, S, O>
where
    S: SemanticStateTransaction,
    O: TransitionOutput,
{
    /// Creates an empty scheduler under validated limits.
    pub fn new(limits: SchedulerLimits) -> Result<Self, Error> {
        let limits = SchedulerLimits::with_execution_bounds(
            limits.max_active_requests,
            limits.max_queued_work,
            limits.max_new_submissions_per_turn,
            limits.max_in_flight_global,
            limits.max_in_flight_per_request,
            limits.execution_slice,
        )?;
        Ok(Self {
            limits,
            requests: BTreeMap::new(),
            terminal: BTreeMap::new(),
            ready: VecDeque::new(),
            prepared: VecDeque::new(),
            in_flight: Vec::new(),
            lifecycle: BTreeMap::new(),
            accepted_work: 0,
            peak_accepted_work: 0,
            peak_in_flight_work: 0,
            submitted_work: 0,
            completed_work: 0,
            failed_work: 0,
            discarded_work: 0,
            cancellation_before_submission: 0,
            cancellation_after_submission: 0,
            abandoned_released_work: 0,
            peak_abandoned_retained_resources: 0,
            last_cancellation_to_release: None,
            max_cancellation_to_release: None,
            finished_requests: 0,
            cancelled_requests: 0,
            deadline_expired_requests: 0,
            drain_cycles: 0,
            observed_backends: BTreeSet::new(),
            all_outputs_preemptible: true,
            poisoned: None,
        })
    }

    /// Registers a unique request and transfers canonical state ownership.
    pub fn register(&mut self, request: RequestId, state: S) -> Result<(), Error> {
        self.validate_registration(request)?;
        self.requests.insert(
            request,
            RequestEntry {
                state,
                pending: VecDeque::new(),
                next_sequence: 0,
            },
        );
        Ok(())
    }

    /// Validates identity and active-request capacity before state allocation.
    pub fn validate_registration(&self, request: RequestId) -> Result<(), Error> {
        self.ensure_ready()?;
        if self.requests.contains_key(&request) || self.terminal.contains_key(&request) {
            return Err(Error::Parallel(format!(
                "request {} is already known to this scheduler",
                request.value()
            )));
        }
        if self.requests.len() >= self.limits.max_active_requests {
            return Err(Error::Parallel(format!(
                "scheduler active-request capacity {} is exhausted",
                self.limits.max_active_requests
            )));
        }
        Ok(())
    }

    /// Returns immutable canonical state for an active request.
    pub fn request_state(&self, request: RequestId) -> Option<&S> {
        self.requests.get(&request).map(|entry| &entry.state)
    }

    /// Returns mutable canonical state only when the request has no branch.
    pub fn request_state_mut(&mut self, request: RequestId) -> Result<&mut S, Error> {
        self.ensure_ready()?;
        if self.branch_count(request) != 0 {
            return Err(Error::Parallel(format!(
                "request {} has prepared or submitted state branches",
                request.value()
            )));
        }
        self.requests
            .get_mut(&request)
            .map(|entry| &mut entry.state)
            .ok_or_else(|| Error::Parallel(format!("request {} is not active", request.value())))
    }

    /// Enqueues one transition without a deadline.
    pub fn enqueue(&mut self, request: RequestId, work: W) -> Result<WorkId, Error> {
        self.enqueue_with_deadline(request, work, None)
    }

    /// Enqueues one transition with an optional absolute deadline.
    pub fn enqueue_with_deadline(
        &mut self,
        request: RequestId,
        work: W,
        deadline: Option<Instant>,
    ) -> Result<WorkId, Error> {
        Ok(self
            .enqueue_batch_with_deadline(request, vec![(work, deadline)])?
            .pop()
            .expect("one enqueued work item"))
    }

    /// Atomically enqueues an ordered batch without deadlines.
    pub fn enqueue_batch(
        &mut self,
        request: RequestId,
        work: Vec<W>,
    ) -> Result<Vec<WorkId>, Error> {
        self.enqueue_batch_with_deadline(
            request,
            work.into_iter().map(|work| (work, None)).collect(),
        )
    }

    fn enqueue_batch_with_deadline(
        &mut self,
        request: RequestId,
        work: Vec<(W, Option<Instant>)>,
    ) -> Result<Vec<WorkId>, Error> {
        self.ensure_ready()?;
        let requested = work.len();
        let accepted_after = self
            .accepted_work
            .checked_add(requested)
            .ok_or_else(|| Error::Parallel("scheduler accepted-work occupancy overflow".into()))?;
        if accepted_after > self.limits.max_queued_work {
            return Err(Error::Parallel(format!(
                "scheduler queue capacity {} cannot accept {requested} items with {} outstanding",
                self.limits.max_queued_work, self.accepted_work
            )));
        }
        let entry = self
            .requests
            .get_mut(&request)
            .ok_or_else(|| Error::Parallel(format!("request {} is not active", request.value())))?;
        let requested_u64 = u64::try_from(requested)
            .map_err(|_| Error::Parallel("work batch length exceeds u64".into()))?;
        let next_sequence = entry
            .next_sequence
            .checked_add(requested_u64)
            .ok_or_else(|| {
                Error::Parallel(format!("request {} exhausted work ids", request.value()))
            })?;
        let was_empty = entry.pending.is_empty();
        let mut ids = Vec::with_capacity(requested);
        for (offset, (work, deadline)) in work.into_iter().enumerate() {
            let id = WorkId::new(request, entry.next_sequence + offset as u64);
            entry.pending.push_back(QueuedWork { id, work, deadline });
            self.lifecycle.insert(id, WorkLifecycle::Queued);
            ids.push(id);
        }
        entry.next_sequence = next_sequence;
        if was_empty && requested != 0 {
            self.ready.push_back(request);
        }
        self.accepted_work = accepted_after;
        self.peak_accepted_work = self.peak_accepted_work.max(accepted_after);
        self.submitted_work = self.submitted_work.saturating_add(requested_u64);
        Ok(ids)
    }

    /// Creates bounded fair descriptors and semantic branches without submission.
    pub fn prepare_bounded(&mut self, max_work: usize, now: Instant) -> Result<usize, Error>
    where
        W: WorkDescriptor,
    {
        self.ensure_ready()?;
        if max_work == 0 {
            return Err(Error::Parallel(
                "scheduler preparation bound must be positive".into(),
            ));
        }
        self.expire_deadlines(now)?;
        let mut prepared = 0;
        let mut stalled = 0;
        while prepared < max_work && !self.ready.is_empty() {
            let request = self
                .ready
                .pop_front()
                .expect("checked nonempty ready queue");
            let branches = self.branch_count(request);
            let can_branch = self
                .requests
                .get(&request)
                .is_some_and(|entry| branches == 0 || entry.state.permits_parallel_branches());
            if !can_branch || branches >= self.limits.max_in_flight_per_request {
                self.ready.push_back(request);
                stalled += 1;
                if stalled >= self.ready.len() {
                    break;
                }
                continue;
            }
            stalled = 0;
            let queued = self
                .requests
                .get_mut(&request)
                .and_then(|entry| entry.pending.pop_front())
                .expect("ready request owns queued work");
            if self
                .requests
                .get(&request)
                .is_some_and(|entry| !entry.pending.is_empty())
            {
                self.ready.push_back(request);
            }
            let mut descriptor = Vec::new();
            let slice_size = queued.work.execution_slice_size();
            if slice_size == 0 || slice_size > self.limits.execution_slice {
                let error = Error::Parallel(format!(
                    "work {:?} execution slice {slice_size} exceeds configured bound {}",
                    queued.id, self.limits.execution_slice
                ));
                self.mark_failed_before_submission(queued.id, error.to_string());
                return Err(error);
            }
            if let Err(error) = queued.work.encode_descriptor(&mut descriptor) {
                self.mark_failed_before_submission(queued.id, error.to_string());
                return Err(error);
            }
            let branch = match self
                .requests
                .get(&request)
                .expect("active request")
                .state
                .branch()
            {
                Ok(branch) => branch,
                Err(error) => {
                    self.mark_failed_before_submission(queued.id, error.to_string());
                    return Err(error);
                }
            };
            self.lifecycle.insert(queued.id, WorkLifecycle::Prepared);
            self.prepared.push_back(PreparedWork {
                id: queued.id,
                work: queued.work,
                descriptor,
                branch,
                deadline: queued.deadline,
            });
            prepared += 1;
        }
        Ok(prepared)
    }

    /// Submits a bounded prepared prefix and attaches each exact completion.
    pub fn submit_prepared(
        &mut self,
        now: Instant,
        mut execute: impl FnMut(WorkId, &W, &mut S::Branch) -> Result<O, Error>,
    ) -> Result<usize, Error> {
        self.ensure_ready()?;
        self.expire_deadlines(now)?;
        let capacity = self
            .limits
            .max_in_flight_global
            .saturating_sub(self.in_flight.len())
            .min(self.limits.max_new_submissions_per_turn);
        let mut submitted = 0;
        while submitted < capacity {
            let Some(mut prepared) = self.prepared.pop_front() else {
                break;
            };
            if prepared.deadline.is_some_and(|deadline| deadline <= now) {
                let request = prepared.id.request();
                self.prepared.push_front(prepared);
                self.cancel_internal(request, CancellationCause::Deadline, now)?;
                continue;
            }
            let output = match execute(prepared.id, &prepared.work, &mut prepared.branch) {
                Ok(output) => output,
                Err(error) => {
                    let id = prepared.id;
                    let discard = S::discard_branch(prepared.branch).err();
                    self.lifecycle.insert(id, WorkLifecycle::Failed);
                    self.failed_work = self.failed_work.saturating_add(1);
                    self.accepted_work = self.accepted_work.saturating_sub(1);
                    self.fail_request(id.request());
                    return match discard {
                        Some(discard) => Err(Error::Parallel(format!(
                            "{error}; branch discard also failed: {discard}"
                        ))),
                        None => Err(error),
                    };
                }
            };
            let backend = output.backend()?;
            self.observed_backends.insert(backend_wire(backend));
            self.all_outputs_preemptible &= output.physically_preemptible();
            self.lifecycle.insert(prepared.id, WorkLifecycle::Submitted);
            self.in_flight.push(SubmittedWork {
                id: prepared.id,
                work: prepared.work,
                branch: prepared.branch,
                output,
                disposition: SubmittedDisposition::Publish,
            });
            submitted += 1;
            self.peak_in_flight_work = self.peak_in_flight_work.max(self.in_flight.len());
        }
        if submitted != 0 {
            self.drain_cycles = self.drain_cycles.saturating_add(1);
        }
        self.update_abandoned_resource_peak();
        Ok(submitted)
    }

    /// Polls exact completions and publishes only successful, non-abandoned work.
    pub fn poll_completions(&mut self, now: Instant) -> SchedulerProgress<W, O> {
        let mut progress = SchedulerProgress::default();
        let mut retained = Vec::with_capacity(self.in_flight.len());
        let submitted_work = std::mem::take(&mut self.in_flight);
        for submitted in submitted_work {
            match submitted.output.is_complete() {
                Ok(false) => retained.push(submitted),
                Ok(true) => self.resolve_completed(submitted, now, &mut progress),
                Err(error) => {
                    let id = submitted.id;
                    let discard_error = S::discard_branch(submitted.branch).err();
                    self.lifecycle.insert(id, WorkLifecycle::Failed);
                    self.failed_work = self.failed_work.saturating_add(1);
                    self.accepted_work = self.accepted_work.saturating_sub(1);
                    self.fail_request(id.request());
                    progress.failed.push(FailedWork {
                        id,
                        error: match discard_error {
                            Some(discard) => {
                                format!("{error}; branch discard also failed: {discard}")
                            }
                            None => error.to_string(),
                        },
                    });
                }
            }
        }
        self.in_flight = retained;
        self.update_abandoned_resource_peak();
        progress
    }

    /// Runs one bounded local scheduling turn.
    pub fn run_local_turn(
        &mut self,
        now: Instant,
        execute: impl FnMut(WorkId, &W, &mut S::Branch) -> Result<O, Error>,
    ) -> Result<SchedulerProgress<W, O>, Error>
    where
        W: WorkDescriptor,
    {
        let mut progress = self.poll_completions(now);
        self.prepare_bounded(self.limits.max_new_submissions_per_turn, now)?;
        progress.newly_submitted = self.submit_prepared(now, execute)?;
        let after_submit = self.poll_completions(now);
        progress.committed.extend(after_submit.committed);
        progress.failed.extend(after_submit.failed);
        Ok(progress)
    }

    /// Runs one bounded distributed turn with exact descriptor and disposition consensus.
    pub fn run_distributed_turn(
        &mut self,
        protocol: u64,
        group: &Group,
        stream: &Stream,
        now: Instant,
        execute: impl FnMut(WorkId, &W, &mut S::Branch) -> Result<O, Error>,
    ) -> Result<SchedulerProgress<W, O>, Error>
    where
        W: WorkDescriptor,
    {
        self.ensure_ready()?;
        self.expire_deadlines_distributed(protocol, group, stream, now)?;
        let mut progress = self.poll_distributed(protocol, group, stream, now)?;
        self.prepare_bounded(self.limits.max_new_submissions_per_turn, now)?;
        let descriptors = self
            .prepared
            .iter()
            .take(
                self.limits
                    .max_in_flight_global
                    .saturating_sub(self.in_flight.len())
                    .min(self.limits.max_new_submissions_per_turn),
            )
            .map(|work| PlannedWork {
                id: work.id,
                descriptor: work.descriptor.clone(),
            })
            .collect::<Vec<_>>();
        if let Err(error) =
            validate_schedule_consensus(&descriptors, self.drain_cycles, protocol, group, stream)
        {
            self.poison(error.to_string());
            return Err(error);
        }
        progress.newly_submitted = match self.submit_prepared(now, execute) {
            Ok(count) => count,
            Err(error) => {
                self.poison(error.to_string());
                return Err(error);
            }
        };
        Ok(progress)
    }

    /// Marks a request complete. The request must have no unpublished work.
    pub fn finish(&mut self, request: RequestId) -> Result<(), Error> {
        self.ensure_ready()?;
        if self
            .in_flight
            .iter()
            .any(|work| work.id.request() == request)
        {
            return Err(Error::Parallel(format!(
                "request {} still has submitted work",
                request.value()
            )));
        }
        let entry = self
            .requests
            .remove(&request)
            .ok_or_else(|| Error::Parallel(format!("request {} is not active", request.value())))?;
        let queued = entry.pending.len();
        for work in entry.pending {
            self.lifecycle.insert(work.id, WorkLifecycle::Abandoned);
        }
        self.ready.retain(|candidate| *candidate != request);
        let (prepared, discard_error) = self.discard_prepared_for_request(request);
        let discarded = queued + prepared;
        self.accepted_work = self.accepted_work.saturating_sub(discarded);
        self.discarded_work = self.discarded_work.saturating_add(discarded as u64);
        self.terminal.insert(
            request,
            if discard_error.is_some() {
                RequestStatus::Failed
            } else {
                RequestStatus::Finished
            },
        );
        self.finished_requests = self.finished_requests.saturating_add(1);
        discard_error.map_or(Ok(()), |error| Err(Error::Parallel(error)))
    }

    /// Cancels locally, abandoning submitted work without waiting for it.
    pub fn cancel(&mut self, request: RequestId) -> Result<(), Error> {
        self.cancel_internal(request, CancellationCause::Explicit, Instant::now())
    }

    /// Reaches exact topology-scoped cancellation consensus before disposition.
    pub fn cancel_distributed(
        &mut self,
        protocol: u64,
        request: RequestId,
        group: &Group,
        stream: &Stream,
    ) -> Result<(), Error> {
        validate_disposition_consensus(
            protocol,
            request,
            CancellationCause::Explicit,
            group,
            stream,
        )
        .inspect_err(|error| self.poison(error.to_string()))?;
        self.cancel_internal(request, CancellationCause::Explicit, Instant::now())
    }

    /// Releases an idle request and returns canonical program state.
    pub fn release(&mut self, request: RequestId) -> Result<S, Error> {
        self.ensure_ready()?;
        let entry = self
            .requests
            .get(&request)
            .ok_or_else(|| Error::Parallel(format!("request {} is not active", request.value())))?;
        if !entry.pending.is_empty() || self.branch_count(request) != 0 {
            return Err(Error::Parallel(format!(
                "request {} still owns unpublished work",
                request.value()
            )));
        }
        Ok(self
            .requests
            .remove(&request)
            .expect("checked request")
            .state)
    }

    /// Removes a terminal identity so a caller may explicitly reuse it.
    pub fn forget_terminal(&mut self, request: RequestId) -> Result<RequestStatus, Error> {
        if self
            .in_flight
            .iter()
            .any(|work| work.id.request() == request)
        {
            return Err(Error::Parallel(format!(
                "request {} still has retained abandoned work",
                request.value()
            )));
        }
        self.terminal
            .remove(&request)
            .ok_or_else(|| Error::Parallel(format!("request {} is not terminal", request.value())))
    }

    /// Returns the known request lifecycle state.
    pub fn request_status(&self, request: RequestId) -> Option<RequestStatus> {
        self.requests
            .contains_key(&request)
            .then_some(RequestStatus::Active)
            .or_else(|| self.terminal.get(&request).copied())
    }

    /// Returns one work item's authoritative lifecycle.
    pub fn work_lifecycle(&self, work: WorkId) -> Option<WorkLifecycle> {
        self.lifecycle.get(&work).copied()
    }

    /// Returns queued transitions for one active request.
    pub fn queued_for_request(&self, request: RequestId) -> usize {
        self.requests
            .get(&request)
            .map_or(0, |entry| entry.pending.len())
    }

    /// Returns static capabilities plus observed backend identity.
    pub fn capabilities(&self) -> SchedulerCapabilities {
        SchedulerCapabilities {
            limits: self.limits,
            observed_backends: self
                .observed_backends
                .iter()
                .copied()
                .map(backend_from_wire)
                .collect(),
            executing_work_physically_preemptible: !self.observed_backends.is_empty()
                && self.all_outputs_preemptible,
            non_preemptible_interval:
                "from exact backend submission until that transition's completion event resolves",
        }
    }

    /// Returns current occupancy and cumulative telemetry.
    pub fn report(&self) -> SchedulerReport {
        let abandoned_in_flight_work = self
            .in_flight
            .iter()
            .filter(|work| matches!(work.disposition, SubmittedDisposition::Abandon { .. }))
            .count();
        let failed_in_flight_work = self
            .in_flight
            .iter()
            .filter(|work| matches!(work.disposition, SubmittedDisposition::Fail))
            .count();
        let submitted_in_flight_work =
            self.in_flight.len() - abandoned_in_flight_work - failed_in_flight_work;
        SchedulerReport {
            active_requests: self.requests.len(),
            queued_work: self
                .requests
                .values()
                .map(|entry| entry.pending.len())
                .sum(),
            prepared_work: self.prepared.len(),
            submitted_in_flight_work,
            completing_work: 0,
            abandoned_in_flight_work,
            failed_in_flight_work,
            current_in_flight_work: self.in_flight.len(),
            peak_in_flight_work: self.peak_in_flight_work,
            peak_queued_work: self.peak_accepted_work,
            submitted_work: self.submitted_work,
            completed_work: self.completed_work,
            failed_work: self.failed_work,
            discarded_work: self.discarded_work,
            cancellation_before_submission: self.cancellation_before_submission,
            cancellation_after_submission: self.cancellation_after_submission,
            abandoned_released_work: self.abandoned_released_work,
            abandoned_retained_resources: self.abandoned_retained_resources(),
            peak_abandoned_retained_resources: self.peak_abandoned_retained_resources,
            last_cancellation_to_release_ns: self
                .last_cancellation_to_release
                .map(|value| value.as_nanos()),
            max_cancellation_to_release_ns: self
                .max_cancellation_to_release
                .map(|value| value.as_nanos()),
            finished_requests: self.finished_requests,
            cancelled_requests: self.cancelled_requests,
            deadline_expired_requests: self.deadline_expired_requests,
            drain_cycles: self.drain_cycles,
            configured_submission_bound: self.limits.max_new_submissions_per_turn,
            configured_slice_bound: self.limits.execution_slice,
            poisoned: self.poisoned.is_some(),
        }
    }

    /// Returns the distributed ordering failure, if any.
    pub fn poison_reason(&self) -> Option<&str> {
        self.poisoned.as_deref()
    }

    fn resolve_completed(
        &mut self,
        submitted: SubmittedWork<W, S::Branch, O>,
        now: Instant,
        progress: &mut SchedulerProgress<W, O>,
    ) {
        match submitted.disposition {
            SubmittedDisposition::Abandon { cancelled_at, .. } => {
                if let Err(error) = S::discard_branch(submitted.branch) {
                    self.lifecycle.insert(submitted.id, WorkLifecycle::Failed);
                    self.failed_work = self.failed_work.saturating_add(1);
                    progress.failed.push(FailedWork {
                        id: submitted.id,
                        error: format!("failed to discard abandoned state branch: {error}"),
                    });
                } else {
                    self.lifecycle
                        .insert(submitted.id, WorkLifecycle::Abandoned);
                }
                self.abandoned_released_work = self.abandoned_released_work.saturating_add(1);
                self.accepted_work = self.accepted_work.saturating_sub(1);
                let latency = now.saturating_duration_since(cancelled_at);
                self.last_cancellation_to_release = Some(latency);
                self.max_cancellation_to_release = Some(
                    self.max_cancellation_to_release
                        .map_or(latency, |previous| previous.max(latency)),
                );
            }
            SubmittedDisposition::Publish => {
                self.lifecycle
                    .insert(submitted.id, WorkLifecycle::Completing);
                let Some(request) = self.requests.get_mut(&submitted.id.request()) else {
                    self.lifecycle
                        .insert(submitted.id, WorkLifecycle::Abandoned);
                    self.accepted_work = self.accepted_work.saturating_sub(1);
                    return;
                };
                if let Err(error) = request.state.commit_branch(submitted.branch) {
                    self.lifecycle.insert(submitted.id, WorkLifecycle::Failed);
                    self.failed_work = self.failed_work.saturating_add(1);
                    self.accepted_work = self.accepted_work.saturating_sub(1);
                    self.fail_request(submitted.id.request());
                    progress.failed.push(FailedWork {
                        id: submitted.id,
                        error: error.to_string(),
                    });
                    return;
                }
                self.lifecycle
                    .insert(submitted.id, WorkLifecycle::Committed);
                self.completed_work = self.completed_work.saturating_add(1);
                self.accepted_work = self.accepted_work.saturating_sub(1);
                progress.committed.push(CompletedWork {
                    id: submitted.id,
                    work: submitted.work,
                    output: submitted.output,
                });
            }
            SubmittedDisposition::Fail => {
                let discard_error = S::discard_branch(submitted.branch).err();
                self.lifecycle.insert(submitted.id, WorkLifecycle::Failed);
                self.accepted_work = self.accepted_work.saturating_sub(1);
                self.fail_request(submitted.id.request());
                progress.failed.push(FailedWork {
                    id: submitted.id,
                    error: discard_error.map_or_else(
                        || "distributed backend completion failed on at least one rank".into(),
                        |discard| format!("distributed backend completion failed on at least one rank; branch discard also failed: {discard}"),
                    ),
                });
            }
        }
    }

    fn poll_distributed(
        &mut self,
        protocol: u64,
        group: &Group,
        stream: &Stream,
        now: Instant,
    ) -> Result<SchedulerProgress<W, O>, Error> {
        let mut local = Vec::with_capacity(self.in_flight.len());
        for work in &self.in_flight {
            let status = match work.output.is_complete() {
                Ok(false) => 0,
                Ok(true) => 1,
                Err(_) => 2,
            };
            local.push((work.id, status));
        }
        let global = validate_completion_consensus(protocol, &local, group, stream)
            .inspect_err(|error| self.poison(error.to_string()))?;
        let mut progress = SchedulerProgress::default();
        let mut retained = Vec::with_capacity(self.in_flight.len());
        let submitted_work = std::mem::take(&mut self.in_flight);
        for (work, status) in submitted_work.into_iter().zip(global) {
            match status {
                0 => retained.push(work),
                1 => self.resolve_completed(work, now, &mut progress),
                3 => {
                    let mut work = work;
                    if !matches!(work.disposition, SubmittedDisposition::Fail) {
                        work.disposition = SubmittedDisposition::Fail;
                        self.lifecycle.insert(work.id, WorkLifecycle::Failed);
                        self.failed_work = self.failed_work.saturating_add(1);
                    }
                    retained.push(work);
                }
                _ => {
                    let id = work.id;
                    if !matches!(work.disposition, SubmittedDisposition::Fail) {
                        self.failed_work = self.failed_work.saturating_add(1);
                    }
                    let discard_error = S::discard_branch(work.branch).err();
                    self.lifecycle.insert(id, WorkLifecycle::Failed);
                    self.accepted_work = self.accepted_work.saturating_sub(1);
                    self.fail_request(id.request());
                    progress.failed.push(FailedWork {
                        id,
                        error: discard_error.map_or_else(
                            || "distributed backend completion failed on at least one rank".into(),
                            |discard| format!("distributed backend completion failed on at least one rank; branch discard also failed: {discard}"),
                        ),
                    });
                }
            }
        }
        self.in_flight = retained;
        Ok(progress)
    }

    fn expire_deadlines(&mut self, now: Instant) -> Result<(), Error> {
        let mut expired = BTreeSet::new();
        for (request, entry) in &self.requests {
            if entry
                .pending
                .iter()
                .any(|work| work.deadline.is_some_and(|deadline| deadline <= now))
            {
                expired.insert(*request);
            }
        }
        for work in &self.prepared {
            if work.deadline.is_some_and(|deadline| deadline <= now) {
                expired.insert(work.id.request());
            }
        }
        for request in expired {
            self.cancel_internal(request, CancellationCause::Deadline, now)?;
        }
        Ok(())
    }

    fn expire_deadlines_distributed(
        &mut self,
        protocol: u64,
        group: &Group,
        stream: &Stream,
        now: Instant,
    ) -> Result<(), Error> {
        let expired = self
            .requests
            .iter()
            .filter_map(|(request, entry)| {
                entry
                    .pending
                    .iter()
                    .any(|work| work.deadline.is_some_and(|deadline| deadline <= now))
                    .then_some(*request)
            })
            .collect::<Vec<_>>();
        for request in expired {
            validate_disposition_consensus(
                protocol,
                request,
                CancellationCause::Deadline,
                group,
                stream,
            )
            .inspect_err(|error| self.poison(error.to_string()))?;
            self.cancel_internal(request, CancellationCause::Deadline, now)?;
        }
        Ok(())
    }

    fn cancel_internal(
        &mut self,
        request: RequestId,
        cause: CancellationCause,
        now: Instant,
    ) -> Result<(), Error> {
        self.ensure_ready()?;
        if self.terminal.contains_key(&request) {
            return Err(Error::Parallel(format!(
                "request {} is already terminal",
                request.value()
            )));
        }
        let entry = self
            .requests
            .remove(&request)
            .ok_or_else(|| Error::Parallel(format!("request {} is not active", request.value())))?;
        let queued = entry.pending.len();
        for work in entry.pending {
            self.lifecycle.insert(work.id, WorkLifecycle::Abandoned);
        }
        self.ready.retain(|candidate| *candidate != request);

        let (prepared, discard_error) = self.discard_prepared_for_request(request);
        let before_submission = queued + prepared;
        self.accepted_work = self.accepted_work.saturating_sub(before_submission);
        self.discarded_work = self.discarded_work.saturating_add(before_submission as u64);
        self.cancellation_before_submission = self
            .cancellation_before_submission
            .saturating_add(before_submission as u64);

        let mut after_submission = 0;
        for work in &mut self.in_flight {
            if work.id.request() == request
                && matches!(work.disposition, SubmittedDisposition::Publish)
            {
                work.disposition = SubmittedDisposition::Abandon {
                    cause,
                    cancelled_at: now,
                };
                self.lifecycle.insert(work.id, WorkLifecycle::Abandoned);
                after_submission += 1;
            }
        }
        self.cancellation_after_submission = self
            .cancellation_after_submission
            .saturating_add(after_submission);
        let status = match cause {
            CancellationCause::Explicit => {
                self.cancelled_requests = self.cancelled_requests.saturating_add(1);
                RequestStatus::Cancelled
            }
            CancellationCause::Deadline => {
                self.deadline_expired_requests = self.deadline_expired_requests.saturating_add(1);
                RequestStatus::DeadlineExceeded
            }
        };
        self.terminal.insert(
            request,
            if discard_error.is_some() {
                RequestStatus::Failed
            } else {
                status
            },
        );
        self.update_abandoned_resource_peak();
        discard_error.map_or(Ok(()), |error| Err(Error::Parallel(error)))
    }

    fn discard_prepared_for_request(&mut self, request: RequestId) -> (usize, Option<String>) {
        let mut retained = VecDeque::with_capacity(self.prepared.len());
        let mut discarded = 0;
        let mut errors = Vec::new();
        for work in std::mem::take(&mut self.prepared) {
            if work.id.request() == request {
                discarded += 1;
                match S::discard_branch(work.branch) {
                    Ok(()) => {
                        self.lifecycle.insert(work.id, WorkLifecycle::Abandoned);
                    }
                    Err(error) => {
                        self.lifecycle.insert(work.id, WorkLifecycle::Failed);
                        self.failed_work = self.failed_work.saturating_add(1);
                        errors.push(format!("work {:?}: {error}", work.id));
                    }
                }
            } else {
                retained.push_back(work);
            }
        }
        self.prepared = retained;
        let error = (!errors.is_empty()).then(|| {
            format!(
                "failed to discard {} prepared state branch(es): {}",
                errors.len(),
                errors.join("; ")
            )
        });
        (discarded, error)
    }

    fn branch_count(&self, request: RequestId) -> usize {
        self.prepared
            .iter()
            .filter(|work| work.id.request() == request)
            .count()
            + self
                .in_flight
                .iter()
                .filter(|work| work.id.request() == request)
                .count()
    }

    fn mark_failed_before_submission(&mut self, id: WorkId, _error: String) {
        self.lifecycle.insert(id, WorkLifecycle::Failed);
        self.failed_work = self.failed_work.saturating_add(1);
        self.accepted_work = self.accepted_work.saturating_sub(1);
        self.fail_request(id.request());
    }

    fn fail_request(&mut self, request: RequestId) {
        let Some(entry) = self.requests.remove(&request) else {
            return;
        };
        let discarded = entry.pending.len();
        for work in entry.pending {
            self.lifecycle.insert(work.id, WorkLifecycle::Failed);
        }
        self.ready.retain(|candidate| *candidate != request);
        let before = self.prepared.len();
        self.prepared.retain(|work| work.id.request() != request);
        let discarded = discarded + before - self.prepared.len();
        self.accepted_work = self.accepted_work.saturating_sub(discarded);
        self.discarded_work = self.discarded_work.saturating_add(discarded as u64);
        for work in &mut self.in_flight {
            if work.id.request() == request
                && matches!(work.disposition, SubmittedDisposition::Publish)
            {
                work.disposition = SubmittedDisposition::Abandon {
                    cause: CancellationCause::Explicit,
                    cancelled_at: Instant::now(),
                };
                self.lifecycle.insert(work.id, WorkLifecycle::Abandoned);
            }
        }
        self.terminal.insert(request, RequestStatus::Failed);
    }

    fn abandoned_retained_resources(&self) -> usize {
        self.in_flight
            .iter()
            .filter(|work| matches!(work.disposition, SubmittedDisposition::Abandon { .. }))
            .map(|work| work.output.retained_resources())
            .sum()
    }

    fn update_abandoned_resource_peak(&mut self) {
        self.peak_abandoned_retained_resources = self
            .peak_abandoned_retained_resources
            .max(self.abandoned_retained_resources());
    }

    fn ensure_ready(&self) -> Result<(), Error> {
        if let Some(reason) = &self.poisoned {
            Err(Error::Parallel(format!(
                "scheduler is poisoned after unsafe distributed ordering: {reason}"
            )))
        } else {
            Ok(())
        }
    }

    fn poison(&mut self, reason: String) {
        if self.poisoned.is_some() {
            return;
        }
        for (request, _) in std::mem::take(&mut self.requests) {
            self.terminal.insert(request, RequestStatus::Failed);
        }
        self.ready.clear();
        self.prepared.clear();
        for work in &mut self.in_flight {
            if matches!(work.disposition, SubmittedDisposition::Publish) {
                work.disposition = SubmittedDisposition::Abandon {
                    cause: CancellationCause::Explicit,
                    cancelled_at: Instant::now(),
                };
            }
        }
        self.poisoned = Some(reason);
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct PlannedWork {
    id: WorkId,
    descriptor: Vec<u32>,
}

fn backend_wire(backend: EventBackend) -> u8 {
    match backend {
        EventBackend::None => 0,
        EventBackend::Cpu => 1,
        EventBackend::Metal => 2,
        EventBackend::Cuda => 3,
    }
}

fn backend_from_wire(value: u8) -> EventBackend {
    match value {
        0 => EventBackend::None,
        1 => EventBackend::Cpu,
        2 => EventBackend::Metal,
        3 => EventBackend::Cuda,
        _ => unreachable!("stored backend wire value is validated"),
    }
}

fn validate_schedule_consensus(
    plan: &[PlannedWork],
    drain_cycle: u64,
    protocol: u64,
    group: &Group,
    stream: &Stream,
) -> Result<(), Error> {
    let mut packed = vec![
        plan.len() as u32,
        drain_cycle as u32,
        (drain_cycle >> 32) as u32,
        protocol as u32,
        (protocol >> 32) as u32,
    ];
    for work in plan {
        push_u64(&mut packed, work.id.request().value());
        push_u64(&mut packed, work.id.sequence());
        packed.push(u32::try_from(work.descriptor.len()).map_err(|_| {
            Error::Parallel("distributed work descriptor length exceeds u32".into())
        })?);
        packed.extend_from_slice(&work.descriptor);
    }
    validate_equal_words(&packed, "distributed schedule", group, stream)
}

fn validate_disposition_consensus(
    protocol: u64,
    request: RequestId,
    cause: CancellationCause,
    group: &Group,
    stream: &Stream,
) -> Result<(), Error> {
    let mut words = vec![protocol as u32, (protocol >> 32) as u32];
    push_u64(&mut words, request.value());
    words.push(match cause {
        CancellationCause::Explicit => 1,
        CancellationCause::Deadline => 2,
    });
    validate_equal_words(
        &words,
        "distributed cancellation disposition",
        group,
        stream,
    )
}

fn validate_completion_consensus(
    protocol: u64,
    local: &[(WorkId, u32)],
    group: &Group,
    stream: &Stream,
) -> Result<Vec<u32>, Error> {
    if group.size() == 1 {
        return Ok(local.iter().map(|(_, status)| *status).collect());
    }
    let mut words = vec![protocol as u32, (protocol >> 32) as u32, local.len() as u32];
    for (id, status) in local {
        push_u64(&mut words, id.request().value());
        push_u64(&mut words, id.sequence());
        words.push(*status);
    }
    let gathered = gather_words(&words, group, stream)?;
    for rank in 0..group.size() {
        let candidate = &gathered[rank * words.len()..(rank + 1) * words.len()];
        if candidate[..3] != words[..3] {
            return Err(Error::Parallel(format!(
                "distributed completion header differs at rank {rank}"
            )));
        }
        for (index, (id, _)) in local.iter().enumerate() {
            let offset = 3 + index * 5;
            let expected = [
                id.request().value() as u32,
                (id.request().value() >> 32) as u32,
                id.sequence() as u32,
                (id.sequence() >> 32) as u32,
            ];
            if candidate[offset..offset + 4] != expected {
                return Err(Error::Parallel(format!(
                    "distributed completion identity differs at rank {rank}"
                )));
            }
            if candidate[offset + 4] > 2 {
                return Err(Error::Parallel(format!(
                    "distributed completion status is invalid at rank {rank}"
                )));
            }
        }
    }
    Ok((0..local.len())
        .map(|index| {
            let statuses = (0..group.size()).map(|rank| {
                let offset = rank * words.len() + 3 + index * 5;
                gathered[offset + 4]
            });
            let statuses = statuses.collect::<Vec<_>>();
            let failed = statuses.contains(&2);
            let incomplete = statuses.contains(&0);
            match (failed, incomplete) {
                (true, true) => 3,
                (true, false) => 2,
                (false, true) => 0,
                (false, false) => 1,
            }
        })
        .collect())
}

fn validate_equal_words(
    words: &[u32],
    context: &str,
    group: &Group,
    stream: &Stream,
) -> Result<(), Error> {
    if group.size() == 1 {
        return Ok(());
    }
    let gathered = gather_words(words, group, stream)?;
    for rank in 0..group.size() {
        let start = rank * words.len();
        let end = start + words.len();
        if gathered.get(start..end) != Some(words) {
            return Err(Error::Parallel(format!("{context} differs at rank {rank}")));
        }
    }
    Ok(())
}

fn gather_words(words: &[u32], group: &Group, stream: &Stream) -> Result<Vec<u32>, Error> {
    let length = i32::try_from(words.len())
        .map_err(|_| Error::Parallel("distributed scheduler metadata exceeds i32".into()))?;
    let local = Array::from_slice(words, &[length]);
    let gathered = distributed::all_gather(&local, group, stream).map_err(|error| {
        Error::Parallel(format!("distributed scheduler consensus failed: {error}"))
    })?;
    async_eval_with_event([&gathered])?.synchronize()?;
    Ok(gathered.evaluated()?.as_slice::<u32>().to_vec())
}

fn push_u64(output: &mut Vec<u32>, value: u64) {
    output.extend_from_slice(&[value as u32, (value >> 32) as u32]);
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        rc::Rc,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };

    use super::*;

    #[derive(Debug, Clone)]
    struct TestState {
        canonical: Vec<u32>,
        prng: u64,
    }

    impl SemanticStateTransaction for TestState {
        type Branch = Self;

        fn branch(&self) -> Result<Self::Branch, Error> {
            Ok(self.clone())
        }

        fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Error> {
            *self = branch;
            Ok(())
        }
    }

    #[derive(Debug)]
    struct RollbackState {
        discarded: Rc<Cell<usize>>,
    }

    impl SemanticStateTransaction for RollbackState {
        type Branch = Rc<Cell<usize>>;

        fn branch(&self) -> Result<Self::Branch, Error> {
            Ok(Rc::clone(&self.discarded))
        }

        fn commit_branch(&mut self, _branch: Self::Branch) -> Result<(), Error> {
            Ok(())
        }

        fn discard_branch(branch: Self::Branch) -> Result<(), Error> {
            branch.set(branch.get() + 1);
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    struct TestWork {
        value: u32,
        descriptor_encodes: Arc<AtomicUsize>,
    }

    impl WorkDescriptor for TestWork {
        fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Error> {
            self.descriptor_encodes.fetch_add(1, Ordering::Relaxed);
            output.push(self.value);
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    struct ControlledCompletion {
        complete: Rc<Cell<bool>>,
        fail: Rc<Cell<bool>>,
        retained: usize,
        output: u32,
    }

    impl TransitionOutput for ControlledCompletion {
        fn is_complete(&self) -> Result<bool, Error> {
            if self.fail.get() {
                Err(Error::Parallel("injected asynchronous failure".into()))
            } else {
                Ok(self.complete.get())
            }
        }

        fn backend(&self) -> Result<EventBackend, Error> {
            Ok(EventBackend::Metal)
        }

        fn retained_resources(&self) -> usize {
            self.retained
        }
    }

    type TestScheduler = FairScheduler<TestWork, TestState, ControlledCompletion>;

    fn scheduler(limits: SchedulerLimits) -> TestScheduler {
        FairScheduler::new(limits).unwrap()
    }

    fn work(value: u32, encodes: &Arc<AtomicUsize>) -> TestWork {
        TestWork {
            value,
            descriptor_encodes: Arc::clone(encodes),
        }
    }

    #[test]
    fn cancel_before_preparation_never_encodes_a_descriptor() {
        let encodes = Arc::new(AtomicUsize::new(0));
        let mut scheduler = scheduler(SchedulerLimits::new(1, 2).unwrap());
        let request = RequestId::new(1);
        scheduler
            .register(
                request,
                TestState {
                    canonical: vec![],
                    prng: 7,
                },
            )
            .unwrap();
        let id = scheduler.enqueue(request, work(1, &encodes)).unwrap();
        scheduler.cancel(request).unwrap();
        assert_eq!(encodes.load(Ordering::Relaxed), 0);
        assert_eq!(scheduler.work_lifecycle(id), Some(WorkLifecycle::Abandoned));
        assert_eq!(scheduler.report().cancellation_before_submission, 1);
    }

    #[test]
    fn cancel_after_preparation_drops_the_branch_without_submission() {
        let encodes = Arc::new(AtomicUsize::new(0));
        let discarded = Rc::new(Cell::new(0));
        let mut scheduler = FairScheduler::<TestWork, RollbackState, ControlledCompletion>::new(
            SchedulerLimits::new(1, 2).unwrap(),
        )
        .unwrap();
        let request = RequestId::new(1);
        scheduler
            .register(
                request,
                RollbackState {
                    discarded: Rc::clone(&discarded),
                },
            )
            .unwrap();
        let id = scheduler.enqueue(request, work(1, &encodes)).unwrap();
        scheduler.prepare_bounded(1, Instant::now()).unwrap();
        assert_eq!(scheduler.work_lifecycle(id), Some(WorkLifecycle::Prepared));
        scheduler.cancel(request).unwrap();
        assert_eq!(scheduler.report().prepared_work, 0);
        assert_eq!(scheduler.report().current_in_flight_work, 0);
        assert_eq!(discarded.get(), 1);
    }

    #[test]
    fn cancel_after_submission_retains_then_releases_without_publication() {
        let encodes = Arc::new(AtomicUsize::new(0));
        let complete = Rc::new(Cell::new(false));
        let mut scheduler = scheduler(SchedulerLimits::new(1, 2).unwrap());
        let request = RequestId::new(1);
        scheduler
            .register(
                request,
                TestState {
                    canonical: vec![],
                    prng: 7,
                },
            )
            .unwrap();
        let id = scheduler.enqueue(request, work(3, &encodes)).unwrap();
        scheduler.prepare_bounded(1, Instant::now()).unwrap();
        scheduler
            .submit_prepared(Instant::now(), |_, work, branch| {
                branch.canonical.push(work.value);
                branch.prng += 1;
                Ok(ControlledCompletion {
                    complete: Rc::clone(&complete),
                    fail: Rc::new(Cell::new(false)),
                    retained: 4,
                    output: work.value,
                })
            })
            .unwrap();
        assert!(scheduler
            .request_state(request)
            .unwrap()
            .canonical
            .is_empty());
        assert_eq!(scheduler.request_state(request).unwrap().prng, 7);
        scheduler.cancel(request).unwrap();
        assert_eq!(scheduler.work_lifecycle(id), Some(WorkLifecycle::Abandoned));
        assert_eq!(scheduler.report().abandoned_retained_resources, 4);
        assert!(scheduler
            .poll_completions(Instant::now())
            .committed
            .is_empty());
        complete.set(true);
        assert!(scheduler
            .poll_completions(Instant::now())
            .committed
            .is_empty());
        assert_eq!(scheduler.report().abandoned_retained_resources, 0);
        assert_eq!(scheduler.report().abandoned_released_work, 1);
    }

    #[test]
    fn completion_racing_cancellation_has_one_authoritative_winner() {
        let encodes = Arc::new(AtomicUsize::new(0));
        let complete = Rc::new(Cell::new(true));
        let mut scheduler = scheduler(SchedulerLimits::new(1, 2).unwrap());
        let request = RequestId::new(1);
        scheduler
            .register(
                request,
                TestState {
                    canonical: vec![],
                    prng: 7,
                },
            )
            .unwrap();
        scheduler.enqueue(request, work(9, &encodes)).unwrap();
        scheduler.prepare_bounded(1, Instant::now()).unwrap();
        scheduler
            .submit_prepared(Instant::now(), |_, work, branch| {
                branch.canonical.push(work.value);
                Ok(ControlledCompletion {
                    complete: Rc::clone(&complete),
                    fail: Rc::new(Cell::new(false)),
                    retained: 1,
                    output: work.value,
                })
            })
            .unwrap();
        let progress = scheduler.poll_completions(Instant::now());
        assert_eq!(progress.committed.len(), 1);
        assert_eq!(scheduler.request_state(request).unwrap().canonical, [9]);
        scheduler.cancel(request).unwrap();
    }

    #[test]
    fn asynchronous_failure_is_request_local() {
        let encodes = Arc::new(AtomicUsize::new(0));
        let fail = Rc::new(Cell::new(false));
        let limits = SchedulerLimits::with_execution_bounds(2, 4, 2, 2, 1, 1).unwrap();
        let mut scheduler = scheduler(limits);
        for request in [RequestId::new(1), RequestId::new(2)] {
            scheduler
                .register(
                    request,
                    TestState {
                        canonical: vec![],
                        prng: 0,
                    },
                )
                .unwrap();
            scheduler
                .enqueue(request, work(request.value() as u32, &encodes))
                .unwrap();
        }
        scheduler.prepare_bounded(2, Instant::now()).unwrap();
        scheduler
            .submit_prepared(Instant::now(), |id, _, _| {
                Ok(ControlledCompletion {
                    complete: Rc::new(Cell::new(id.request() == RequestId::new(2))),
                    fail: if id.request() == RequestId::new(1) {
                        Rc::clone(&fail)
                    } else {
                        Rc::new(Cell::new(false))
                    },
                    retained: 1,
                    output: 0,
                })
            })
            .unwrap();
        fail.set(true);
        let progress = scheduler.poll_completions(Instant::now());
        assert_eq!(progress.failed.len(), 1);
        assert_eq!(progress.committed.len(), 1);
        assert_eq!(
            scheduler.request_status(RequestId::new(1)),
            Some(RequestStatus::Failed)
        );
        assert_eq!(
            scheduler.request_status(RequestId::new(2)),
            Some(RequestStatus::Active)
        );
        assert!(!scheduler.report().poisoned);
    }

    #[test]
    fn deadline_expiry_cancels_before_submission() {
        let encodes = Arc::new(AtomicUsize::new(0));
        let mut scheduler = scheduler(SchedulerLimits::new(1, 2).unwrap());
        let request = RequestId::new(1);
        scheduler
            .register(
                request,
                TestState {
                    canonical: vec![],
                    prng: 0,
                },
            )
            .unwrap();
        let now = Instant::now();
        scheduler
            .enqueue_with_deadline(request, work(1, &encodes), Some(now))
            .unwrap();
        scheduler.prepare_bounded(1, now).unwrap();
        assert_eq!(
            scheduler.request_status(request),
            Some(RequestStatus::DeadlineExceeded)
        );
        assert_eq!(encodes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn execution_slice_bound_rejects_oversized_work_before_encoding() {
        #[derive(Debug)]
        struct SlicedWork(Arc<AtomicUsize>);

        impl WorkDescriptor for SlicedWork {
            fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Error> {
                self.0.fetch_add(1, Ordering::Relaxed);
                output.push(1);
                Ok(())
            }

            fn execution_slice_size(&self) -> usize {
                2
            }
        }

        let encodes = Arc::new(AtomicUsize::new(0));
        let limits = SchedulerLimits::with_execution_bounds(1, 2, 1, 1, 1, 1).unwrap();
        let mut scheduler =
            FairScheduler::<SlicedWork, TestState, ControlledCompletion>::new(limits).unwrap();
        let request = RequestId::new(1);
        scheduler
            .register(
                request,
                TestState {
                    canonical: vec![],
                    prng: 0,
                },
            )
            .unwrap();
        scheduler
            .enqueue(request, SlicedWork(Arc::clone(&encodes)))
            .unwrap();
        let error = scheduler.prepare_bounded(1, Instant::now()).unwrap_err();
        assert!(error.to_string().contains("execution slice 2"));
        assert_eq!(encodes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn global_and_per_request_in_flight_bounds_preserve_fairness() {
        let encodes = Arc::new(AtomicUsize::new(0));
        let limits = SchedulerLimits::with_execution_bounds(2, 8, 2, 2, 1, 1).unwrap();
        let mut scheduler = scheduler(limits);
        for request in [RequestId::new(1), RequestId::new(2)] {
            scheduler
                .register(
                    request,
                    TestState {
                        canonical: vec![],
                        prng: 0,
                    },
                )
                .unwrap();
            scheduler.enqueue(request, work(1, &encodes)).unwrap();
            scheduler.enqueue(request, work(2, &encodes)).unwrap();
        }
        scheduler.prepare_bounded(4, Instant::now()).unwrap();
        assert_eq!(scheduler.report().prepared_work, 2);
        let slow = Rc::new(Cell::new(false));
        scheduler
            .submit_prepared(Instant::now(), |id, _, _| {
                Ok(ControlledCompletion {
                    complete: if id.request() == RequestId::new(1) {
                        Rc::clone(&slow)
                    } else {
                        Rc::new(Cell::new(true))
                    },
                    fail: Rc::new(Cell::new(false)),
                    retained: 1,
                    output: id.request().value() as u32,
                })
            })
            .unwrap();
        let progress = scheduler.poll_completions(Instant::now());
        assert_eq!(progress.committed[0].output.output, 2);
        scheduler.prepare_bounded(2, Instant::now()).unwrap();
        assert_eq!(scheduler.report().prepared_work, 1);
        assert_eq!(scheduler.report().peak_in_flight_work, 2);
    }

    #[test]
    fn distributed_cancellation_reaches_single_rank_consensus_before_disposition() {
        use safemlx::{distributed::Backend, Device, DeviceType};

        let group = Group::init(false, Backend::Any).unwrap();
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let encodes = Arc::new(AtomicUsize::new(0));
        let mut scheduler = scheduler(SchedulerLimits::new(1, 2).unwrap());
        let request = RequestId::new(77);
        scheduler
            .register(
                request,
                TestState {
                    canonical: vec![],
                    prng: 0,
                },
            )
            .unwrap();
        scheduler.enqueue(request, work(1, &encodes)).unwrap();
        scheduler
            .cancel_distributed(0xCA11_CE11, request, &group, &stream)
            .unwrap();
        assert_eq!(
            scheduler.request_status(request),
            Some(RequestStatus::Cancelled)
        );
        assert_eq!(encodes.load(Ordering::Relaxed), 0);
    }

    struct MetalCompletion {
        event: safemlx::Event,
        retained: Vec<Array>,
    }

    impl TransitionOutput for MetalCompletion {
        fn is_complete(&self) -> Result<bool, Error> {
            Ok(self.event.is_complete()?)
        }

        fn backend(&self) -> Result<EventBackend, Error> {
            Ok(self.event.backend()?)
        }

        fn retained_resources(&self) -> usize {
            self.retained.len()
        }
    }

    #[test]
    #[ignore = "explicit Metal cancellation test; run outside the sandbox on a Metal host"]
    fn metal_abandonment_is_nonblocking_retained_and_submission_bounded() {
        use safemlx::{transforms::async_eval_with_event, Device, DeviceType};

        let producer = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let unrelated = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let unrelated_lhs = Array::ones::<f32>(&[4096, 4096], &unrelated).unwrap();
        let unrelated_rhs = Array::ones::<f32>(&[4096, 4096], &unrelated).unwrap();
        let unrelated_output = unrelated_lhs.matmul(&unrelated_rhs, &unrelated).unwrap();
        let unrelated_event = async_eval_with_event([&unrelated_output]).unwrap();

        let encodes = Arc::new(AtomicUsize::new(0));
        let limits = SchedulerLimits::with_execution_bounds(2, 4, 1, 1, 1, 1).unwrap();
        let mut scheduler =
            FairScheduler::<TestWork, TestState, MetalCompletion>::new(limits).unwrap();
        let cancelled = RequestId::new(1);
        let waiting = RequestId::new(2);
        for request in [cancelled, waiting] {
            scheduler
                .register(
                    request,
                    TestState {
                        canonical: vec![],
                        prng: 41,
                    },
                )
                .unwrap();
            scheduler.enqueue(request, work(1, &encodes)).unwrap();
        }
        let progress = scheduler
            .run_local_turn(Instant::now(), |_, work, branch| {
                branch.canonical.push(work.value);
                branch.prng += 1;
                let lhs = Array::ones::<f32>(&[4096, 4096], &producer)?;
                let rhs = Array::ones::<f32>(&[4096, 4096], &producer)?;
                let output = lhs.matmul(&rhs, &producer)?;
                let event = async_eval_with_event([&output])?;
                Ok(MetalCompletion {
                    event,
                    retained: vec![lhs, rhs, output],
                })
            })
            .unwrap();
        assert_eq!(progress.newly_submitted, 1);
        assert_eq!(scheduler.report().current_in_flight_work, 1);
        assert_eq!(scheduler.report().queued_work, 1);
        assert!(scheduler
            .request_state(cancelled)
            .unwrap()
            .canonical
            .is_empty());
        assert_eq!(scheduler.request_state(cancelled).unwrap().prng, 41);

        scheduler.cancel(cancelled).unwrap();
        assert!(scheduler.report().abandoned_retained_resources >= 3);
        assert!(
            !unrelated_event.is_complete().unwrap(),
            "cancellation must return before unrelated stream work is drained"
        );
        while scheduler.report().abandoned_in_flight_work != 0 {
            scheduler.poll_completions(Instant::now());
            std::thread::yield_now();
        }
        assert_eq!(scheduler.report().abandoned_retained_resources, 0);
        assert_eq!(scheduler.report().abandoned_released_work, 1);
        assert_eq!(
            scheduler.capabilities().observed_backends,
            [EventBackend::Metal]
        );
        assert!(
            !scheduler
                .capabilities()
                .executing_work_physically_preemptible
        );
        unrelated_event.synchronize().unwrap();
    }
}

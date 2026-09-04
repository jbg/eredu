//! Fair transactional scheduling independent of event and tensor runtimes.

use crate::consensus::{
    agree_deadline_candidates_bounded, agree_disposition_status_bounded, resolve_completions,
    validate_schedule, BoundedConsensusTransport, CompletionObservation, CompletionResolution,
    ConsensusTransport, ScheduledWork,
};
use crate::BoundedCompletionWait;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    time::{Duration, Instant},
};

/// Stable caller-assigned request identity.
#[derive(Debug, Clone, Copy, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
pub struct RequestId(u64);

impl RequestId {
    /// Creates an identity. Zero is valid.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns its numeric value.
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// Scheduler-assigned ordered transition identity.
#[derive(Debug, Clone, Copy, Eq, Ord, PartialEq, PartialOrd, Hash, Serialize, Deserialize)]
pub struct WorkId {
    request: RequestId,
    sequence: u64,
}

impl WorkId {
    /// Creates an ordered transition identity.
    pub const fn new(request: RequestId, sequence: u64) -> Self {
        Self { request, sequence }
    }

    /// Request updated by this work.
    pub const fn request(self) -> RequestId {
        self.request
    }

    /// Zero-based transition number.
    pub const fn sequence(self) -> u64 {
        self.sequence
    }
}

/// Authoritative lifecycle of one transition.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkLifecycle {
    /// Accepted but not branched.
    Queued,
    /// Branch exists; nothing submitted.
    Prepared,
    /// Exact completion exists and is incomplete.
    Submitted,
    /// Completion succeeded and commit is resolving.
    Completing,
    /// Branch was published.
    Committed,
    /// Submitted work was cancelled and is retained until exact completion.
    Abandoned,
    /// Preparation, execution, completion, or commit failed.
    Failed,
}

/// Observable lifecycle of a request.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestStatus {
    /// Owns canonical state.
    Active,
    /// Finished normally.
    Finished,
    /// Explicitly cancelled.
    Cancelled,
    /// Deadline expired.
    DeadlineExceeded,
    /// Request-local failure.
    Failed,
}

/// Cause attached to cancellation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancellationCause {
    /// Caller cancellation.
    Explicit,
    /// Deadline expiry.
    Deadline,
}

/// Capacity and cooperative-preemption controls.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct SchedulerLimits {
    /// Maximum active requests.
    pub max_active_requests: usize,
    /// Maximum accepted nonterminal work.
    pub max_queued_work: usize,
    /// Maximum submissions per scheduling turn.
    pub max_new_submissions_per_turn: usize,
    /// Maximum global in-flight work.
    pub max_in_flight_global: usize,
    /// Maximum per-request in-flight work.
    pub max_in_flight_per_request: usize,
    /// Program-defined operations per transition.
    pub execution_slice: usize,
}

impl SchedulerLimits {
    /// Creates conservative bounds.
    pub fn new(max_active_requests: usize, max_queued_work: usize) -> Result<Self, SchedulerError> {
        Self::with_execution_bounds(
            max_active_requests,
            max_queued_work,
            1,
            max_active_requests,
            1,
            usize::MAX,
        )
    }

    /// Creates explicit bounds.
    pub fn with_execution_bounds(
        max_active_requests: usize,
        max_queued_work: usize,
        max_new_submissions_per_turn: usize,
        max_in_flight_global: usize,
        max_in_flight_per_request: usize,
        execution_slice: usize,
    ) -> Result<Self, SchedulerError> {
        let values = [
            max_active_requests,
            max_queued_work,
            max_new_submissions_per_turn,
            max_in_flight_global,
            max_in_flight_per_request,
            execution_slice,
        ];
        if values.contains(&0) {
            return Err(SchedulerError::InvalidLimits(values));
        }
        Ok(Self {
            max_active_requests,
            max_queued_work,
            max_new_submissions_per_turn,
            max_in_flight_global,
            max_in_flight_per_request,
            execution_slice,
        })
    }
}

impl Default for SchedulerLimits {
    fn default() -> Self {
        Self {
            max_active_requests: 64,
            max_queued_work: 256,
            max_new_submissions_per_turn: 1,
            max_in_flight_global: 64,
            max_in_flight_per_request: 1,
            execution_slice: usize::MAX,
        }
    }
}

/// Stable semantic description of one work item.
pub trait WorkDescriptor {
    /// Descriptor construction error.
    type Error: std::error::Error;

    /// Appends every semantic field in stable wire order.
    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Self::Error>;

    /// Number of program-defined non-preemptible operations in this transition.
    fn execution_slice_size(&self) -> usize {
        1
    }
}

/// Transaction over canonical semantic state.
pub trait SemanticStateTransaction {
    /// Transition-local branch.
    type Branch;
    /// State error.
    type Error: std::error::Error;

    /// Creates an unpublished branch.
    fn branch(&self) -> Result<Self::Branch, Self::Error>;

    /// Publishes a completed branch.
    fn commit_branch(&mut self, branch: Self::Branch) -> Result<(), Self::Error>;

    /// Discards an unpublished branch.
    fn discard_branch(branch: Self::Branch) -> Result<(), Self::Error> {
        drop(branch);
        Ok(())
    }

    /// Whether independent branches may share the current canonical base.
    fn permits_parallel_branches(&self) -> bool {
        false
    }
}

/// Exact backend completion retaining submitted resources.
pub trait TransitionOutput {
    /// Completion observation error.
    type Error: std::error::Error;

    /// Nonblocking exact completion observation.
    fn is_complete(&self) -> Result<bool, Self::Error>;

    /// Stable backend name used only for capability telemetry.
    fn backend_name(&self) -> Option<String> {
        None
    }

    /// Whether already executing work can be physically interrupted.
    fn physically_preemptible(&self) -> bool {
        false
    }

    /// Explicitly retained resource count.
    fn retained_resources(&self) -> usize;
}

#[derive(Debug)]
struct Request<W, S> {
    state: S,
    next: u64,
    pending: VecDeque<Queued<W>>,
}

#[derive(Debug)]
struct Queued<W> {
    id: WorkId,
    work: W,
    deadline: Option<Instant>,
}

#[derive(Debug)]
struct Prepared<W, B> {
    id: WorkId,
    work: W,
    descriptor: Vec<u32>,
    branch: B,
    deadline: Option<Instant>,
}

#[derive(Debug, Clone, Copy)]
enum Disposition {
    Publish,
    Abandon { cancelled_at: Instant },
    Fail,
}

#[derive(Debug)]
struct Submitted<W, B, O> {
    id: WorkId,
    work: W,
    branch: B,
    output: O,
    disposition: Disposition,
}

/// Result of one scheduling turn.
#[derive(Debug)]
pub struct SchedulerProgress<W, O> {
    /// Newly submitted transitions.
    pub newly_submitted: usize,
    /// Successfully committed work and backend outputs.
    pub committed: Vec<(WorkId, W, O)>,
    /// Failed work with structured scheduler context.
    pub failed: Vec<(WorkId, SchedulerError)>,
}

impl<W, O> Default for SchedulerProgress<W, O> {
    fn default() -> Self {
        Self {
            newly_submitted: 0,
            committed: Vec::new(),
            failed: Vec::new(),
        }
    }
}

/// Static scheduler capabilities and configured bounds.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct SchedulerCapabilities {
    /// Configured scheduler limits.
    pub limits: SchedulerLimits,
    /// Backends observed on submitted transitions.
    pub observed_backends: Vec<String>,
    /// Whether every observed backend output supports physical preemption.
    pub executing_work_physically_preemptible: bool,
    /// Exact description of the unavoidable cancellation interval.
    pub non_preemptible_interval: String,
}

/// Snapshot of scheduler occupancy and cumulative telemetry.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct SchedulerReport {
    /// Active requests.
    pub active_requests: usize,
    /// Queued work.
    pub queued_work: usize,
    /// Prepared work.
    pub prepared_work: usize,
    /// Submitted work still eligible to publish.
    pub submitted_in_flight_work: usize,
    /// Work currently resolving publication.
    pub completing_work: usize,
    /// Cancelled submitted work awaiting exact completion.
    pub abandoned_in_flight_work: usize,
    /// Globally failed work retained until every rank reaches completion.
    pub failed_in_flight_work: usize,
    /// All submitted work, including abandoned work.
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
    /// Work cancelled before submission.
    pub cancellation_before_submission: u64,
    /// Work cancelled after submission.
    pub cancellation_after_submission: u64,
    /// Abandoned work released after exact completion.
    pub abandoned_released_work: u64,
    /// Resources retained by abandoned work.
    pub abandoned_retained_resources: usize,
    /// Maximum resources retained by abandoned work.
    pub peak_abandoned_retained_resources: usize,
    /// Last cancellation-to-release latency in nanoseconds.
    pub last_cancellation_to_release_ns: Option<u128>,
    /// Maximum cancellation-to-release latency in nanoseconds.
    pub max_cancellation_to_release_ns: Option<u128>,
    /// Requests ended normally.
    pub finished_requests: u64,
    /// Requests cancelled explicitly.
    pub cancelled_requests: u64,
    /// Requests ended by deadline.
    pub deadline_expired_requests: u64,
    /// Turns which submitted at least one transition.
    pub drain_cycles: u64,
    /// Configured per-turn submission bound.
    pub configured_submission_bound: usize,
    /// Configured execution-slice bound.
    pub configured_slice_bound: usize,
    /// Whether unsafe distributed ordering poisoned the scheduler.
    pub poisoned: bool,
}

/// Fair scheduler whose state transitions are independent of backend objects.
#[derive(Debug)]
pub struct Scheduler<W, S: SemanticStateTransaction, O: TransitionOutput> {
    limits: SchedulerLimits,
    requests: BTreeMap<RequestId, Request<W, S>>,
    terminal: BTreeMap<RequestId, RequestStatus>,
    ready: VecDeque<RequestId>,
    prepared: VecDeque<Prepared<W, S::Branch>>,
    submitted: Vec<Submitted<W, S::Branch, O>>,
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
    observed_backends: BTreeSet<String>,
    all_outputs_preemptible: bool,
    poisoned: Option<String>,
}

impl<W, S: SemanticStateTransaction, O: TransitionOutput> Scheduler<W, S, O> {
    /// Creates an empty scheduler under validated limits.
    pub fn new(limits: SchedulerLimits) -> Result<Self, SchedulerError> {
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
            submitted: Vec::new(),
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

    /// Validates identity and active-request capacity before allocating state.
    pub fn validate_registration(&self, id: RequestId) -> Result<(), SchedulerError> {
        self.ensure_ready()?;
        if self.requests.contains_key(&id) || self.terminal.contains_key(&id) {
            return Err(SchedulerError::DuplicateRequest(id));
        }
        if self.requests.len() >= self.limits.max_active_requests {
            return Err(SchedulerError::Capacity(format!(
                "scheduler active-request capacity {} is exhausted",
                self.limits.max_active_requests
            )));
        }
        Ok(())
    }

    /// Registers canonical request state.
    pub fn register(&mut self, id: RequestId, state: S) -> Result<(), SchedulerError> {
        self.validate_registration(id)?;
        self.requests.insert(
            id,
            Request {
                state,
                next: 0,
                pending: VecDeque::new(),
            },
        );
        Ok(())
    }

    /// Returns immutable canonical state for an active request.
    pub fn request_state(&self, id: RequestId) -> Option<&S> {
        self.requests.get(&id).map(|entry| &entry.state)
    }

    /// Returns mutable canonical state only while no branch exists.
    pub fn request_state_mut(&mut self, id: RequestId) -> Result<&mut S, SchedulerError> {
        self.ensure_ready()?;
        if self.branch_count(id) != 0 {
            return Err(SchedulerError::State(format!(
                "request {} has prepared or submitted state branches",
                id.value()
            )));
        }
        self.requests
            .get_mut(&id)
            .map(|entry| &mut entry.state)
            .ok_or(SchedulerError::UnknownRequest(id))
    }

    /// Enqueues one transition without a deadline.
    pub fn enqueue(&mut self, request: RequestId, work: W) -> Result<WorkId, SchedulerError> {
        self.enqueue_with_deadline(request, work, None)
    }

    /// Enqueues one transition with an optional absolute deadline.
    pub fn enqueue_with_deadline(
        &mut self,
        request: RequestId,
        work: W,
        deadline: Option<Instant>,
    ) -> Result<WorkId, SchedulerError> {
        Ok(self
            .enqueue_batch_with_deadlines(request, vec![(work, deadline)])?
            .pop()
            .expect("one work item was supplied"))
    }

    /// Atomically enqueues an ordered batch.
    pub fn enqueue_batch(
        &mut self,
        request: RequestId,
        work: Vec<W>,
    ) -> Result<Vec<WorkId>, SchedulerError> {
        self.enqueue_batch_with_deadlines(
            request,
            work.into_iter().map(|work| (work, None)).collect(),
        )
    }

    fn enqueue_batch_with_deadlines(
        &mut self,
        request: RequestId,
        work: Vec<(W, Option<Instant>)>,
    ) -> Result<Vec<WorkId>, SchedulerError> {
        self.ensure_ready()?;
        let requested = work.len();
        let accepted_after = self
            .accepted_work
            .checked_add(requested)
            .ok_or_else(|| SchedulerError::Capacity("scheduler occupancy overflow".into()))?;
        if accepted_after > self.limits.max_queued_work {
            return Err(SchedulerError::Capacity(format!(
                "scheduler queue capacity {} cannot accept {requested} items with {} outstanding",
                self.limits.max_queued_work, self.accepted_work
            )));
        }
        let entry = self
            .requests
            .get_mut(&request)
            .ok_or(SchedulerError::UnknownRequest(request))?;
        let count = u64::try_from(requested)
            .map_err(|_| SchedulerError::Capacity("work batch length exceeds u64".into()))?;
        let next = entry
            .next
            .checked_add(count)
            .ok_or_else(|| SchedulerError::Capacity("work identity space exhausted".into()))?;
        let was_empty = entry.pending.is_empty();
        let mut ids = Vec::with_capacity(requested);
        for (offset, (work, deadline)) in work.into_iter().enumerate() {
            let id = WorkId::new(request, entry.next + offset as u64);
            entry.pending.push_back(Queued { id, work, deadline });
            self.lifecycle.insert(id, WorkLifecycle::Queued);
            ids.push(id);
        }
        entry.next = next;
        if was_empty && requested != 0 {
            self.ready.push_back(request);
        }
        self.accepted_work = accepted_after;
        self.peak_accepted_work = self.peak_accepted_work.max(accepted_after);
        self.submitted_work = self.submitted_work.saturating_add(count);
        Ok(ids)
    }

    /// Prepares bounded branches in round-robin request order.
    pub fn prepare_bounded(&mut self, limit: usize, now: Instant) -> Result<usize, SchedulerError>
    where
        W: WorkDescriptor,
    {
        self.ensure_ready()?;
        if limit == 0 {
            return Err(SchedulerError::Capacity(
                "scheduler preparation bound must be positive".into(),
            ));
        }
        self.expire_deadlines(now)?;
        let mut count = 0;
        let mut stalled = 0;
        while count < limit && !self.ready.is_empty() {
            let request = self.ready.pop_front().expect("ready queue is nonempty");
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
            let slice = queued.work.execution_slice_size();
            if slice == 0 || slice > self.limits.execution_slice {
                self.fail_before_submission(queued.id);
                return Err(SchedulerError::Descriptor(format!(
                    "work {:?} execution slice {slice} exceeds configured bound {}",
                    queued.id, self.limits.execution_slice
                )));
            }
            let mut descriptor = Vec::new();
            if let Err(error) = queued.work.encode_descriptor(&mut descriptor) {
                self.fail_before_submission(queued.id);
                return Err(SchedulerError::Descriptor(error.to_string()));
            }
            let branch = match self
                .requests
                .get(&request)
                .expect("active request exists")
                .state
                .branch()
            {
                Ok(branch) => branch,
                Err(error) => {
                    self.fail_before_submission(queued.id);
                    return Err(SchedulerError::State(error.to_string()));
                }
            };
            self.lifecycle.insert(queued.id, WorkLifecycle::Prepared);
            self.prepared.push_back(Prepared {
                id: queued.id,
                work: queued.work,
                descriptor,
                branch,
                deadline: queued.deadline,
            });
            count += 1;
        }
        Ok(count)
    }

    /// Submits a bounded prepared prefix through the injected backend adapter.
    pub fn submit_prepared<E>(
        &mut self,
        now: Instant,
        mut execute: impl FnMut(WorkId, &W, &mut S::Branch) -> Result<O, E>,
    ) -> Result<usize, SchedulerError>
    where
        E: std::error::Error,
    {
        self.ensure_ready()?;
        self.expire_deadlines(now)?;
        let capacity = self
            .limits
            .max_in_flight_global
            .saturating_sub(self.submitted.len())
            .min(self.limits.max_new_submissions_per_turn);
        let mut count = 0;
        while count < capacity {
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
                    let message = discard.map_or_else(
                        || error.to_string(),
                        |discard| format!("{error}; branch discard also failed: {discard}"),
                    );
                    return Err(SchedulerError::Submission(message));
                }
            };
            if let Some(backend) = output.backend_name() {
                self.observed_backends.insert(backend);
            }
            self.all_outputs_preemptible &= output.physically_preemptible();
            self.lifecycle.insert(prepared.id, WorkLifecycle::Submitted);
            self.submitted.push(Submitted {
                id: prepared.id,
                work: prepared.work,
                branch: prepared.branch,
                output,
                disposition: Disposition::Publish,
            });
            count += 1;
            self.peak_in_flight_work = self.peak_in_flight_work.max(self.submitted.len());
        }
        if count != 0 {
            self.drain_cycles = self.drain_cycles.saturating_add(1);
        }
        self.update_abandoned_resource_peak();
        Ok(count)
    }

    /// Polls exact completions and publishes only successful work.
    pub fn poll_completions(&mut self, now: Instant) -> SchedulerProgress<W, O> {
        let mut progress = SchedulerProgress::default();
        let mut retained = Vec::with_capacity(self.submitted.len());
        for submitted in std::mem::take(&mut self.submitted) {
            match submitted.output.is_complete() {
                Ok(false) => retained.push(submitted),
                Ok(true) => self.resolve_completed(submitted, now, &mut progress),
                Err(error) => {
                    let id = submitted.id;
                    let already_failed = matches!(submitted.disposition, Disposition::Fail);
                    let discard = S::discard_branch(submitted.branch).err();
                    self.lifecycle.insert(id, WorkLifecycle::Failed);
                    if !already_failed {
                        self.failed_work = self.failed_work.saturating_add(1);
                    }
                    self.accepted_work = self.accepted_work.saturating_sub(1);
                    self.fail_request(id.request());
                    progress.failed.push((
                        id,
                        SchedulerError::Completion(discard.map_or_else(
                            || error.to_string(),
                            |discard| format!("{error}; branch discard also failed: {discard}"),
                        )),
                    ));
                }
            }
        }
        // A completion failure can make its whole request terminal while sibling
        // submissions from the same poll batch are still incomplete. Those
        // siblings remain backend-owned until their exact completions resolve,
        // but can no longer publish into the removed canonical state.
        for submitted in &mut retained {
            if !self.requests.contains_key(&submitted.id.request())
                && matches!(submitted.disposition, Disposition::Publish)
            {
                submitted.disposition = Disposition::Abandon { cancelled_at: now };
                self.lifecycle
                    .insert(submitted.id, WorkLifecycle::Abandoned);
            }
        }
        self.submitted = retained;
        self.update_abandoned_resource_peak();
        progress
    }

    /// Runs one bounded local scheduling turn.
    pub fn run_local_turn<E>(
        &mut self,
        now: Instant,
        execute: impl FnMut(WorkId, &W, &mut S::Branch) -> Result<O, E>,
    ) -> Result<SchedulerProgress<W, O>, SchedulerError>
    where
        W: WorkDescriptor,
        E: std::error::Error,
    {
        self.ensure_ready()?;
        let mut progress = self.poll_completions(now);
        self.prepare_bounded(self.limits.max_new_submissions_per_turn, now)?;
        progress.newly_submitted = self.submit_prepared(now, execute)?;
        let after_submit = self.poll_completions(now);
        progress.committed.extend(after_submit.committed);
        progress.failed.extend(after_submit.failed);
        Ok(progress)
    }

    /// Runs one bounded turn with topology-wide schedule and completion consensus.
    pub fn run_distributed_turn<T, E>(
        &mut self,
        protocol: u64,
        transport: &T,
        wait: BoundedCompletionWait,
        now: Instant,
        execute: impl FnMut(WorkId, &W, &mut S::Branch) -> Result<O, E>,
    ) -> Result<SchedulerProgress<W, O>, SchedulerError>
    where
        W: WorkDescriptor,
        T: BoundedConsensusTransport,
        <T::Completion as crate::Completion>::Error: std::fmt::Display,
        E: std::error::Error,
    {
        self.ensure_ready()?;
        self.expire_deadlines_distributed(protocol, transport, wait, now)?;
        let mut progress = self.poll_distributed(protocol, transport, now)?;
        self.prepare_bounded(self.limits.max_new_submissions_per_turn, now)?;
        let plan = self
            .prepared
            .iter()
            .take(
                self.limits
                    .max_in_flight_global
                    .saturating_sub(self.submitted.len())
                    .min(self.limits.max_new_submissions_per_turn),
            )
            .map(|work| ScheduledWork {
                id: work.id,
                descriptor: &work.descriptor,
            })
            .collect::<Vec<_>>();
        if let Err(error) = validate_schedule(transport, &plan, self.drain_cycles, protocol) {
            self.poison(error.to_string(), now);
            return Err(SchedulerError::Consensus(error.to_string()));
        }
        progress.newly_submitted = match self.submit_prepared(now, execute) {
            Ok(count) => count,
            Err(error) => {
                self.poison(error.to_string(), now);
                return Err(error);
            }
        };
        Ok(progress)
    }

    /// Reaches topology-wide consensus before cancelling a request.
    pub fn cancel_distributed<T: BoundedConsensusTransport>(
        &mut self,
        protocol: u64,
        request: RequestId,
        transport: &T,
        wait: BoundedCompletionWait,
        now: Instant,
    ) -> Result<(), SchedulerError>
    where
        <T::Completion as crate::Completion>::Error: std::fmt::Display,
    {
        self.ensure_ready()?;
        let locally_ready =
            self.requests.contains_key(&request) && !self.terminal.contains_key(&request);
        let prepared = agree_disposition_status_bounded(
            transport,
            protocol,
            request,
            CancellationCause::Explicit,
            1,
            locally_ready,
            wait,
        );
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                self.fence(error.to_string());
                return Err(SchedulerError::Consensus(error.to_string()));
            }
        };
        if !prepared {
            let error = "distributed cancellation preparation failed on at least one rank";
            self.fence(error.to_string());
            return Err(SchedulerError::Consensus(error.into()));
        }
        let committed = agree_disposition_status_bounded(
            transport,
            protocol,
            request,
            CancellationCause::Explicit,
            2,
            true,
            wait,
        );
        let committed = match committed {
            Ok(committed) => committed,
            Err(error) => {
                self.fence(error.to_string());
                return Err(SchedulerError::Consensus(error.to_string()));
            }
        };
        if !committed {
            let error = "distributed cancellation commit authorization failed";
            self.fence(error.to_string());
            return Err(SchedulerError::Consensus(error.into()));
        }
        let result = self.cancel_internal(request, CancellationCause::Explicit, now);
        if let Err(error) = &result {
            // The disposition is already globally authorized and local
            // cancellation marks every owned work item abandoned before
            // reporting branch-cleanup failure. Fence later scheduler work.
            self.poison(error.to_string(), now);
        }
        result
    }

    /// Marks a request finished and discards its queued work.
    pub fn finish(&mut self, request: RequestId) -> Result<(), SchedulerError> {
        self.ensure_ready()?;
        if self
            .submitted
            .iter()
            .any(|work| work.id.request() == request)
        {
            return Err(SchedulerError::State(format!(
                "request {} still has submitted work",
                request.value()
            )));
        }
        let entry = self
            .requests
            .remove(&request)
            .ok_or(SchedulerError::UnknownRequest(request))?;
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
        discard_error.map_or(Ok(()), |error| Err(SchedulerError::State(error)))
    }

    /// Cancels a request, retaining submitted resources until exact completion.
    pub fn cancel(&mut self, request: RequestId) -> Result<(), SchedulerError> {
        self.cancel_internal(request, CancellationCause::Explicit, Instant::now())
    }

    /// Releases an idle request and returns its canonical state.
    pub fn release(&mut self, request: RequestId) -> Result<S, SchedulerError> {
        self.ensure_ready()?;
        let entry = self
            .requests
            .get(&request)
            .ok_or(SchedulerError::UnknownRequest(request))?;
        if !entry.pending.is_empty() || self.branch_count(request) != 0 {
            return Err(SchedulerError::State(format!(
                "request {} still owns unpublished work",
                request.value()
            )));
        }
        Ok(self
            .requests
            .remove(&request)
            .expect("checked active request")
            .state)
    }

    /// Removes a terminal identity so it may be reused explicitly.
    pub fn forget_terminal(&mut self, request: RequestId) -> Result<RequestStatus, SchedulerError> {
        self.ensure_ready()?;
        if self
            .submitted
            .iter()
            .any(|work| work.id.request() == request)
        {
            return Err(SchedulerError::State(format!(
                "request {} still has retained abandoned work",
                request.value()
            )));
        }
        self.terminal
            .remove(&request)
            .ok_or(SchedulerError::UnknownRequest(request))
    }

    /// Returns active or terminal request status.
    pub fn request_status(&self, id: RequestId) -> Option<RequestStatus> {
        self.requests
            .contains_key(&id)
            .then_some(RequestStatus::Active)
            .or_else(|| self.terminal.get(&id).copied())
    }

    /// Returns a work lifecycle state.
    pub fn work_lifecycle(&self, id: WorkId) -> Option<WorkLifecycle> {
        self.lifecycle.get(&id).copied()
    }

    /// Returns queued transitions for one active request.
    pub fn queued_for_request(&self, request: RequestId) -> usize {
        self.requests
            .get(&request)
            .map_or(0, |entry| entry.pending.len())
    }

    /// Returns configured bounds and observed backend capabilities.
    pub fn capabilities(&self) -> SchedulerCapabilities {
        SchedulerCapabilities {
            limits: self.limits,
            observed_backends: self.observed_backends.iter().cloned().collect(),
            executing_work_physically_preemptible: !self.observed_backends.is_empty()
                && self.all_outputs_preemptible,
            non_preemptible_interval:
                "from exact backend submission until that transition's completion resolves".into(),
        }
    }

    /// Returns current occupancy and cumulative telemetry.
    pub fn report(&self) -> SchedulerReport {
        let abandoned = self
            .submitted
            .iter()
            .filter(|work| matches!(work.disposition, Disposition::Abandon { .. }))
            .count();
        let failed = self
            .submitted
            .iter()
            .filter(|work| matches!(work.disposition, Disposition::Fail))
            .count();
        SchedulerReport {
            active_requests: self.requests.len(),
            queued_work: self
                .requests
                .values()
                .map(|entry| entry.pending.len())
                .sum(),
            prepared_work: self.prepared.len(),
            submitted_in_flight_work: self.submitted.len() - abandoned - failed,
            completing_work: 0,
            abandoned_in_flight_work: abandoned,
            failed_in_flight_work: failed,
            current_in_flight_work: self.submitted.len(),
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
                .map(|duration| duration.as_nanos()),
            max_cancellation_to_release_ns: self
                .max_cancellation_to_release
                .map(|duration| duration.as_nanos()),
            finished_requests: self.finished_requests,
            cancelled_requests: self.cancelled_requests,
            deadline_expired_requests: self.deadline_expired_requests,
            drain_cycles: self.drain_cycles,
            configured_submission_bound: self.limits.max_new_submissions_per_turn,
            configured_slice_bound: self.limits.execution_slice,
            poisoned: self.poisoned.is_some(),
        }
    }

    /// Returns the unsafe distributed-ordering failure, if one occurred.
    pub fn poison_reason(&self) -> Option<&str> {
        self.poisoned.as_deref()
    }

    fn resolve_completed(
        &mut self,
        submitted: Submitted<W, S::Branch, O>,
        now: Instant,
        progress: &mut SchedulerProgress<W, O>,
    ) {
        match submitted.disposition {
            Disposition::Abandon { cancelled_at } => {
                if let Err(error) = S::discard_branch(submitted.branch) {
                    self.lifecycle.insert(submitted.id, WorkLifecycle::Failed);
                    self.failed_work = self.failed_work.saturating_add(1);
                    progress.failed.push((
                        submitted.id,
                        SchedulerError::State(format!(
                            "failed to discard abandoned state branch: {error}"
                        )),
                    ));
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
            Disposition::Publish => {
                self.lifecycle
                    .insert(submitted.id, WorkLifecycle::Completing);
                let Some(request) = self.requests.get_mut(&submitted.id.request()) else {
                    if let Err(error) = S::discard_branch(submitted.branch) {
                        self.lifecycle.insert(submitted.id, WorkLifecycle::Failed);
                        self.failed_work = self.failed_work.saturating_add(1);
                        progress.failed.push((
                            submitted.id,
                            SchedulerError::State(format!(
                                "failed to discard unpublished state branch: {error}"
                            )),
                        ));
                    } else {
                        self.lifecycle
                            .insert(submitted.id, WorkLifecycle::Abandoned);
                    }
                    self.accepted_work = self.accepted_work.saturating_sub(1);
                    return;
                };
                if let Err(error) = request.state.commit_branch(submitted.branch) {
                    self.lifecycle.insert(submitted.id, WorkLifecycle::Failed);
                    self.failed_work = self.failed_work.saturating_add(1);
                    self.accepted_work = self.accepted_work.saturating_sub(1);
                    self.fail_request(submitted.id.request());
                    progress
                        .failed
                        .push((submitted.id, SchedulerError::State(error.to_string())));
                    return;
                }
                self.lifecycle
                    .insert(submitted.id, WorkLifecycle::Committed);
                self.completed_work = self.completed_work.saturating_add(1);
                self.accepted_work = self.accepted_work.saturating_sub(1);
                progress
                    .committed
                    .push((submitted.id, submitted.work, submitted.output));
            }
            Disposition::Fail => {
                let discard = S::discard_branch(submitted.branch).err();
                self.lifecycle.insert(submitted.id, WorkLifecycle::Failed);
                self.accepted_work = self.accepted_work.saturating_sub(1);
                self.fail_request(submitted.id.request());
                progress.failed.push((
                    submitted.id,
                    SchedulerError::DistributedCompletion(discard.map_or_else(
                        || "backend completion failed on at least one rank".into(),
                        |error| {
                            format!(
                                "backend completion failed on at least one rank; branch discard also failed: {error}"
                            )
                        },
                    )),
                ));
            }
        }
    }

    fn poll_distributed<T: ConsensusTransport>(
        &mut self,
        protocol: u64,
        transport: &T,
        now: Instant,
    ) -> Result<SchedulerProgress<W, O>, SchedulerError> {
        let local = self
            .submitted
            .iter()
            .map(|work| {
                let status = match work.output.is_complete() {
                    Ok(false) => CompletionObservation::Incomplete,
                    Ok(true) => CompletionObservation::Complete,
                    Err(_) => CompletionObservation::Failed,
                };
                (work.id, status)
            })
            .collect::<Vec<_>>();
        let global = match resolve_completions(transport, protocol, &local) {
            Ok(global) => global,
            Err(error) => {
                self.poison(error.to_string(), now);
                return Err(SchedulerError::Consensus(error.to_string()));
            }
        };

        let mut progress = SchedulerProgress::default();
        let mut retained = Vec::with_capacity(self.submitted.len());
        for (mut work, status) in std::mem::take(&mut self.submitted).into_iter().zip(global) {
            match status {
                CompletionResolution::Incomplete => retained.push(work),
                CompletionResolution::Complete => {
                    self.resolve_completed(work, now, &mut progress);
                }
                CompletionResolution::FailedPending => {
                    if !matches!(work.disposition, Disposition::Fail) {
                        work.disposition = Disposition::Fail;
                        self.lifecycle.insert(work.id, WorkLifecycle::Failed);
                        self.failed_work = self.failed_work.saturating_add(1);
                    }
                    retained.push(work);
                }
                CompletionResolution::FailedComplete => {
                    if !matches!(work.disposition, Disposition::Fail) {
                        work.disposition = Disposition::Fail;
                        self.failed_work = self.failed_work.saturating_add(1);
                    }
                    self.resolve_completed(work, now, &mut progress);
                }
            }
        }
        for work in &mut retained {
            if !self.requests.contains_key(&work.id.request())
                && matches!(work.disposition, Disposition::Publish)
            {
                work.disposition = Disposition::Abandon { cancelled_at: now };
                self.lifecycle.insert(work.id, WorkLifecycle::Abandoned);
            }
        }
        self.submitted = retained;
        self.update_abandoned_resource_peak();
        Ok(progress)
    }

    fn expire_deadlines(&mut self, now: Instant) -> Result<(), SchedulerError> {
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

    fn expire_deadlines_distributed<T: BoundedConsensusTransport>(
        &mut self,
        protocol: u64,
        transport: &T,
        wait: BoundedCompletionWait,
        now: Instant,
    ) -> Result<(), SchedulerError>
    where
        <T::Completion as crate::Completion>::Error: std::fmt::Display,
    {
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
        let local = self
            .requests
            .keys()
            .copied()
            .map(|request| (request, expired.contains(&request)))
            .collect::<Vec<_>>();
        let globally_expired = match agree_deadline_candidates_bounded(
            transport,
            protocol,
            &local,
            self.limits.max_active_requests,
            wait,
        ) {
            Ok(expired) => expired,
            Err(error) => {
                self.fence(error.to_string());
                return Err(SchedulerError::Consensus(error.to_string()));
            }
        };
        for request in globally_expired {
            for phase in [1, 2] {
                let agreed = match agree_disposition_status_bounded(
                    transport,
                    protocol,
                    request,
                    CancellationCause::Deadline,
                    phase,
                    self.requests.contains_key(&request),
                    wait,
                ) {
                    Ok(agreed) => agreed,
                    Err(error) => {
                        self.fence(error.to_string());
                        return Err(SchedulerError::Consensus(error.to_string()));
                    }
                };
                if !agreed {
                    let error = if phase == 1 {
                        "distributed deadline preparation failed on at least one rank"
                    } else {
                        "distributed deadline commit authorization failed"
                    };
                    self.fence(error.into());
                    return Err(SchedulerError::Consensus(error.into()));
                }
            }
            if let Err(error) = self.cancel_internal(request, CancellationCause::Deadline, now) {
                self.poison(error.to_string(), now);
                return Err(error);
            }
        }
        Ok(())
    }

    fn cancel_internal(
        &mut self,
        request: RequestId,
        cause: CancellationCause,
        now: Instant,
    ) -> Result<(), SchedulerError> {
        self.ensure_ready()?;
        if self.terminal.contains_key(&request) {
            return Err(SchedulerError::State(format!(
                "request {} is already terminal",
                request.value()
            )));
        }
        let entry = self
            .requests
            .remove(&request)
            .ok_or(SchedulerError::UnknownRequest(request))?;
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
        let mut after_submission = 0u64;
        for work in &mut self.submitted {
            if work.id.request() == request && matches!(work.disposition, Disposition::Publish) {
                work.disposition = Disposition::Abandon { cancelled_at: now };
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
        discard_error.map_or(Ok(()), |error| Err(SchedulerError::State(error)))
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
        let error = (!errors.is_empty()).then(|| errors.join("; "));
        (discarded, error)
    }

    fn branch_count(&self, request: RequestId) -> usize {
        self.prepared
            .iter()
            .filter(|work| work.id.request() == request)
            .count()
            + self
                .submitted
                .iter()
                .filter(|work| work.id.request() == request)
                .count()
    }

    fn fail_before_submission(&mut self, id: WorkId) {
        self.lifecycle.insert(id, WorkLifecycle::Failed);
        self.failed_work = self.failed_work.saturating_add(1);
        self.accepted_work = self.accepted_work.saturating_sub(1);
        self.fail_request(id.request());
    }

    fn fail_request(&mut self, request: RequestId) {
        let Some(entry) = self.requests.remove(&request) else {
            return;
        };
        let queued = entry.pending.len();
        for work in entry.pending {
            self.lifecycle.insert(work.id, WorkLifecycle::Failed);
        }
        self.ready.retain(|candidate| *candidate != request);
        let (prepared, _) = self.discard_prepared_for_request(request);
        let discarded = queued + prepared;
        self.accepted_work = self.accepted_work.saturating_sub(discarded);
        self.discarded_work = self.discarded_work.saturating_add(discarded as u64);
        for work in &mut self.submitted {
            if work.id.request() == request && matches!(work.disposition, Disposition::Publish) {
                work.disposition = Disposition::Abandon {
                    cancelled_at: Instant::now(),
                };
                self.lifecycle.insert(work.id, WorkLifecycle::Abandoned);
            }
        }
        self.terminal.insert(request, RequestStatus::Failed);
    }

    fn ensure_ready(&self) -> Result<(), SchedulerError> {
        self.poisoned.as_ref().map_or(Ok(()), |reason| {
            Err(SchedulerError::Poisoned(reason.clone()))
        })
    }

    fn fence(&mut self, reason: String) {
        if self.poisoned.is_none() {
            self.poisoned = Some(reason);
        }
    }

    fn poison(&mut self, reason: String, now: Instant) {
        if self.poisoned.is_some() {
            return;
        }
        let mut discarded = 0usize;
        for (request, entry) in std::mem::take(&mut self.requests) {
            self.terminal.insert(request, RequestStatus::Failed);
            for work in entry.pending {
                self.lifecycle.insert(work.id, WorkLifecycle::Failed);
                discarded += 1;
            }
        }
        self.ready.clear();
        let mut cleanup_errors = Vec::new();
        for work in std::mem::take(&mut self.prepared) {
            let id = work.id;
            if let Err(error) = S::discard_branch(work.branch) {
                cleanup_errors.push(format!("work {id:?}: {error}"));
                self.failed_work = self.failed_work.saturating_add(1);
            }
            self.lifecycle.insert(id, WorkLifecycle::Failed);
            discarded += 1;
        }
        self.accepted_work = self.accepted_work.saturating_sub(discarded);
        self.discarded_work = self.discarded_work.saturating_add(discarded as u64);
        for work in &mut self.submitted {
            if matches!(work.disposition, Disposition::Publish) {
                work.disposition = Disposition::Abandon { cancelled_at: now };
                self.lifecycle.insert(work.id, WorkLifecycle::Abandoned);
            }
        }
        self.poisoned = Some(if cleanup_errors.is_empty() {
            reason
        } else {
            format!(
                "{reason}; prepared branch cleanup failed: {}",
                cleanup_errors.join("; ")
            )
        });
        self.update_abandoned_resource_peak();
    }

    fn abandoned_retained_resources(&self) -> usize {
        self.submitted
            .iter()
            .filter(|work| matches!(work.disposition, Disposition::Abandon { .. }))
            .map(|work| work.output.retained_resources())
            .sum()
    }

    fn update_abandoned_resource_peak(&mut self) {
        self.peak_abandoned_retained_resources = self
            .peak_abandoned_retained_resources
            .max(self.abandoned_retained_resources());
    }
}

/// Structured neutral scheduler error.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum SchedulerError {
    /// One or more limits were zero.
    #[error("scheduler limits must be positive, got {0:?}")]
    InvalidLimits([usize; 6]),
    /// Duplicate request identity.
    #[error("request {} is already registered", .0.value())]
    DuplicateRequest(RequestId),
    /// Unknown request identity.
    #[error("request {} is not active", .0.value())]
    UnknownRequest(RequestId),
    /// Capacity exhausted with contextual detail.
    #[error("{0}")]
    Capacity(String),
    /// Work descriptor validation failed.
    #[error("work descriptor failed: {0}")]
    Descriptor(String),
    /// Semantic state operation failed.
    #[error("semantic state transaction failed: {0}")]
    State(String),
    /// Backend submission failed.
    #[error("backend submission failed: {0}")]
    Submission(String),
    /// Exact completion observation failed.
    #[error("exact completion observation failed: {0}")]
    Completion(String),
    /// Topology-wide scheduler agreement failed.
    #[error("distributed scheduler consensus failed: {0}")]
    Consensus(String),
    /// A distributed backend failed and every rank reached a terminal completion.
    #[error("distributed {0}")]
    DistributedCompletion(String),
    /// Unsafe distributed ordering prevents further scheduler mutation.
    #[error("scheduler is poisoned after unsafe distributed ordering: {0}")]
    Poisoned(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        cell::{Cell, RefCell},
        collections::VecDeque,
        convert::Infallible,
        rc::Rc,
    };

    #[derive(Default)]
    struct State(u32);

    impl SemanticStateTransaction for State {
        type Branch = u32;
        type Error = Infallible;

        fn branch(&self) -> Result<u32, Infallible> {
            Ok(self.0 + 1)
        }

        fn commit_branch(&mut self, branch: u32) -> Result<(), Infallible> {
            self.0 = branch;
            Ok(())
        }

        fn permits_parallel_branches(&self) -> bool {
            true
        }
    }

    impl WorkDescriptor for u32 {
        type Error = Infallible;

        fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Self::Error> {
            output.push(*self);
            Ok(())
        }
    }

    #[derive(Debug)]
    struct Output {
        complete: Rc<Cell<bool>>,
        fail: bool,
    }

    impl TransitionOutput for Output {
        type Error = std::io::Error;

        fn is_complete(&self) -> Result<bool, Self::Error> {
            if self.fail {
                Err(std::io::Error::other("mock failure"))
            } else {
                Ok(self.complete.get())
            }
        }

        fn backend_name(&self) -> Option<String> {
            Some("mock".into())
        }

        fn retained_resources(&self) -> usize {
            2
        }
    }

    #[derive(Default)]
    struct GatherStep {
        replacements: Vec<(usize, usize, u32)>,
    }

    struct ScriptedTransport {
        participants: usize,
        steps: RefCell<VecDeque<GatherStep>>,
    }

    impl ScriptedTransport {
        fn new(participants: usize, steps: Vec<GatherStep>) -> Self {
            Self {
                participants,
                steps: RefCell::new(steps.into()),
            }
        }
    }

    impl ConsensusTransport for ScriptedTransport {
        type Error = Infallible;

        fn participant_count(&self) -> usize {
            self.participants
        }

        fn all_gather_words(&self, local: &[u32]) -> Result<Vec<u32>, Self::Error> {
            let mut gathered = local.repeat(self.participants);
            let step = self.steps.borrow_mut().pop_front().unwrap_or_default();
            for (rank, offset, value) in step.replacements {
                gathered[rank * local.len() + offset] = value;
            }
            Ok(gathered)
        }
    }

    struct ReadyConsensusCompletion;

    impl crate::Completion for ReadyConsensusCompletion {
        type Error = Infallible;

        fn is_complete(&self) -> Result<bool, Self::Error> {
            Ok(true)
        }

        fn wait(&self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    impl crate::BoundedCompletion for ReadyConsensusCompletion {
        fn wait_bounded(
            self,
            _: crate::BoundedCompletionWait,
        ) -> Result<crate::BoundedCompletionOutcome, Self::Error> {
            Ok(crate::BoundedCompletionOutcome::Completed)
        }
    }

    impl BoundedConsensusTransport for ScriptedTransport {
        type Completion = ReadyConsensusCompletion;
        type GatherOutput = Vec<u32>;

        fn submit_all_gather_words(
            &self,
            local: &[u32],
        ) -> Result<crate::Submission<Self::GatherOutput, Self::Completion>, Self::Error> {
            Ok(crate::Submission {
                output: self.all_gather_words(local)?,
                completion: ReadyConsensusCompletion,
            })
        }

        fn resolve_all_gather_words(
            &self,
            output: Self::GatherOutput,
        ) -> Result<Vec<u32>, Self::Error> {
            Ok(output)
        }
    }

    fn consensus_wait() -> crate::BoundedCompletionWait {
        crate::BoundedCompletionWait::new(
            Duration::from_millis(10),
            crate::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap()
    }

    fn scheduler() -> Scheduler<u32, State, Output> {
        Scheduler::new(SchedulerLimits::default()).unwrap()
    }

    #[test]
    fn scheduler_construction_revalidates_public_limit_fields() {
        let invalid = SchedulerLimits {
            max_active_requests: 0,
            ..SchedulerLimits::default()
        };
        assert!(matches!(
            Scheduler::<u32, State, Output>::new(invalid),
            Err(SchedulerError::InvalidLimits(_))
        ));
    }

    #[test]
    fn queued_prepared_submitted_committed_exactly() {
        let done = Rc::new(Cell::new(false));
        let mut scheduler = scheduler();
        let request = RequestId::new(1);
        scheduler.register(request, State::default()).unwrap();
        let id = scheduler.enqueue(request, 7).unwrap();
        assert_eq!(scheduler.work_lifecycle(id), Some(WorkLifecycle::Queued));
        scheduler.prepare_bounded(1, Instant::now()).unwrap();
        assert_eq!(scheduler.work_lifecycle(id), Some(WorkLifecycle::Prepared));
        scheduler
            .submit_prepared(Instant::now(), |_, _, _| {
                Ok::<_, Infallible>(Output {
                    complete: done.clone(),
                    fail: false,
                })
            })
            .unwrap();
        assert!(scheduler
            .poll_completions(Instant::now())
            .committed
            .is_empty());
        done.set(true);
        assert_eq!(
            scheduler.poll_completions(Instant::now()).committed.len(),
            1
        );
        assert_eq!(scheduler.request_state(request).unwrap().0, 1);
    }

    #[test]
    fn cancellation_retains_submitted_resources_until_completion() {
        let done = Rc::new(Cell::new(false));
        let mut scheduler = scheduler();
        let request = RequestId::new(2);
        scheduler.register(request, State::default()).unwrap();
        let id = scheduler.enqueue(request, 1).unwrap();
        scheduler.prepare_bounded(1, Instant::now()).unwrap();
        scheduler
            .submit_prepared(Instant::now(), |_, _, _| {
                Ok::<_, Infallible>(Output {
                    complete: done.clone(),
                    fail: false,
                })
            })
            .unwrap();
        scheduler.cancel(request).unwrap();
        assert_eq!(scheduler.work_lifecycle(id), Some(WorkLifecycle::Abandoned));
        assert_eq!(scheduler.report().abandoned_retained_resources, 2);
        assert_eq!(
            scheduler.request_status(request),
            Some(RequestStatus::Cancelled)
        );
        done.set(true);
        scheduler.poll_completions(Instant::now());
        assert_eq!(scheduler.report().current_in_flight_work, 0);
        assert_eq!(scheduler.report().abandoned_released_work, 1);
    }

    #[test]
    fn exact_completion_failure_does_not_commit() {
        let mut scheduler = scheduler();
        let request = RequestId::new(3);
        scheduler.register(request, State::default()).unwrap();
        let id = scheduler.enqueue(request, 1).unwrap();
        scheduler.prepare_bounded(1, Instant::now()).unwrap();
        scheduler
            .submit_prepared(Instant::now(), |_, _, _| {
                Ok::<_, Infallible>(Output {
                    complete: Rc::new(Cell::new(true)),
                    fail: true,
                })
            })
            .unwrap();
        assert_eq!(scheduler.poll_completions(Instant::now()).failed.len(), 1);
        assert_eq!(scheduler.work_lifecycle(id), Some(WorkLifecycle::Failed));
        assert_eq!(
            scheduler.request_status(request),
            Some(RequestStatus::Failed)
        );
    }

    #[test]
    fn request_failure_abandons_incomplete_sibling_until_exact_completion() {
        let waiting = Rc::new(Cell::new(false));
        let limits = SchedulerLimits::with_execution_bounds(1, 2, 2, 2, 2, 1).unwrap();
        let mut scheduler = Scheduler::new(limits).unwrap();
        let request = RequestId::new(31);
        scheduler.register(request, State::default()).unwrap();
        let ids = scheduler.enqueue_batch(request, vec![1, 2]).unwrap();
        scheduler.prepare_bounded(2, Instant::now()).unwrap();
        scheduler
            .submit_prepared(Instant::now(), |id, _, _| {
                Ok::<_, Infallible>(Output {
                    complete: if id.sequence() == 0 {
                        Rc::new(Cell::new(true))
                    } else {
                        waiting.clone()
                    },
                    fail: id.sequence() == 0,
                })
            })
            .unwrap();

        let progress = scheduler.poll_completions(Instant::now());
        assert_eq!(progress.failed.len(), 1);
        assert_eq!(
            scheduler.work_lifecycle(ids[0]),
            Some(WorkLifecycle::Failed)
        );
        assert_eq!(
            scheduler.work_lifecycle(ids[1]),
            Some(WorkLifecycle::Abandoned)
        );
        assert_eq!(scheduler.report().abandoned_retained_resources, 2);

        waiting.set(true);
        scheduler.poll_completions(Instant::now());
        assert_eq!(scheduler.report().current_in_flight_work, 0);
        assert_eq!(scheduler.report().abandoned_released_work, 1);
    }

    #[test]
    fn submission_failure_marks_work_and_request_failed() {
        let mut scheduler = scheduler();
        let request = RequestId::new(4);
        scheduler.register(request, State::default()).unwrap();
        let id = scheduler.enqueue(request, 1).unwrap();
        scheduler.prepare_bounded(1, Instant::now()).unwrap();
        let error = scheduler
            .submit_prepared(Instant::now(), |_, _, _| {
                Err::<Output, _>(std::io::Error::other("mock submit"))
            })
            .unwrap_err();
        assert!(matches!(error, SchedulerError::Submission(_)));
        assert_eq!(scheduler.work_lifecycle(id), Some(WorkLifecycle::Failed));
        assert_eq!(
            scheduler.request_status(request),
            Some(RequestStatus::Failed)
        );
    }

    #[test]
    fn deadlines_batches_and_fairness_are_backend_neutral() {
        let mut scheduler = scheduler();
        let first = RequestId::new(10);
        let second = RequestId::new(20);
        scheduler.register(first, State::default()).unwrap();
        scheduler.register(second, State::default()).unwrap();
        scheduler.enqueue_batch(first, vec![1, 2]).unwrap();
        scheduler.enqueue(second, 3).unwrap();
        scheduler.prepare_bounded(3, Instant::now()).unwrap();
        let mut order = Vec::new();
        scheduler
            .submit_prepared(Instant::now(), |id, _, _| {
                order.push(id.request());
                Ok::<_, Infallible>(Output {
                    complete: Rc::new(Cell::new(true)),
                    fail: false,
                })
            })
            .unwrap();
        assert_eq!(order, vec![first]);

        let expired = RequestId::new(30);
        scheduler.register(expired, State::default()).unwrap();
        scheduler
            .enqueue_with_deadline(
                expired,
                4,
                Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
            )
            .unwrap();
        scheduler.prepare_bounded(1, Instant::now()).unwrap();
        assert_eq!(
            scheduler.request_status(expired),
            Some(RequestStatus::DeadlineExceeded)
        );
    }

    #[test]
    fn distributed_schedule_mismatch_poisons_the_canonical_machine() {
        let transport = ScriptedTransport::new(
            2,
            vec![
                GatherStep::default(),
                GatherStep::default(),
                GatherStep {
                    replacements: vec![(1, 10, 99)],
                },
            ],
        );
        let mut scheduler = scheduler();
        let request = RequestId::new(50);
        scheduler.register(request, State::default()).unwrap();
        let work = scheduler.enqueue(request, 7).unwrap();

        let error = scheduler
            .run_distributed_turn(
                0xAA,
                &transport,
                consensus_wait(),
                Instant::now(),
                |_, _, _| {
                    Ok::<_, Infallible>(Output {
                        complete: Rc::new(Cell::new(false)),
                        fail: false,
                    })
                },
            )
            .unwrap_err();
        assert!(matches!(error, SchedulerError::Consensus(_)));
        assert!(scheduler.poison_reason().is_some());
        assert!(scheduler.report().poisoned);
        assert_eq!(
            scheduler.request_status(request),
            Some(RequestStatus::Failed)
        );
        assert_eq!(scheduler.work_lifecycle(work), Some(WorkLifecycle::Failed));
        assert!(matches!(
            scheduler.enqueue(request, 8),
            Err(SchedulerError::Poisoned(_))
        ));
    }

    #[test]
    fn poisoned_submissions_remain_retained_until_local_exact_completion() {
        let transport = ScriptedTransport::new(
            2,
            vec![
                GatherStep::default(),
                GatherStep::default(),
                GatherStep::default(),
                GatherStep::default(),
                GatherStep {
                    replacements: vec![(1, 4, 99)],
                },
            ],
        );
        let done = Rc::new(Cell::new(false));
        let mut scheduler = scheduler();
        let request = RequestId::new(54);
        scheduler.register(request, State::default()).unwrap();
        scheduler.enqueue(request, 1).unwrap();
        scheduler
            .run_distributed_turn(
                0xEE,
                &transport,
                consensus_wait(),
                Instant::now(),
                |_, _, _| {
                    Ok::<_, Infallible>(Output {
                        complete: done.clone(),
                        fail: false,
                    })
                },
            )
            .unwrap();

        let error = scheduler
            .run_distributed_turn(
                0xEE,
                &transport,
                consensus_wait(),
                Instant::now(),
                |_, _, _| {
                    Ok::<_, Infallible>(Output {
                        complete: done.clone(),
                        fail: false,
                    })
                },
            )
            .unwrap_err();
        assert!(matches!(error, SchedulerError::Consensus(_)));
        assert_eq!(scheduler.report().abandoned_in_flight_work, 1);
        assert_eq!(scheduler.report().abandoned_retained_resources, 2);

        assert!(scheduler
            .poll_completions(Instant::now())
            .committed
            .is_empty());
        assert_eq!(scheduler.report().current_in_flight_work, 1);
        done.set(true);
        assert!(scheduler
            .poll_completions(Instant::now())
            .committed
            .is_empty());
        assert_eq!(scheduler.report().current_in_flight_work, 0);
        assert_eq!(scheduler.report().abandoned_released_work, 1);
    }

    #[test]
    fn distributed_failure_retains_output_until_every_rank_is_terminal() {
        let transport = ScriptedTransport::new(
            3,
            vec![
                GatherStep::default(),
                GatherStep::default(),
                GatherStep::default(),
                GatherStep::default(),
                GatherStep {
                    replacements: vec![(1, 7, 2), (2, 7, 0)],
                },
                GatherStep::default(),
                GatherStep::default(),
                GatherStep {
                    replacements: vec![(1, 7, 2), (2, 7, 1)],
                },
                GatherStep::default(),
            ],
        );
        let mut scheduler = scheduler();
        let request = RequestId::new(51);
        scheduler.register(request, State::default()).unwrap();
        scheduler.enqueue(request, 1).unwrap();
        let output = || Output {
            complete: Rc::new(Cell::new(true)),
            fail: false,
        };

        scheduler
            .run_distributed_turn(
                0xBB,
                &transport,
                consensus_wait(),
                Instant::now(),
                |_, _, _| {
                    Ok::<_, Infallible>(Output {
                        complete: Rc::new(Cell::new(true)),
                        fail: false,
                    })
                },
            )
            .unwrap();
        let pending = scheduler
            .run_distributed_turn(
                0xBB,
                &transport,
                consensus_wait(),
                Instant::now(),
                |_, _, _| Ok::<_, Infallible>(output()),
            )
            .unwrap();
        assert!(pending.failed.is_empty());
        assert_eq!(scheduler.report().failed_in_flight_work, 1);
        assert_eq!(scheduler.report().current_in_flight_work, 1);

        let terminal = scheduler
            .run_distributed_turn(
                0xBB,
                &transport,
                consensus_wait(),
                Instant::now(),
                |_, _, _| Ok::<_, Infallible>(output()),
            )
            .unwrap();
        assert_eq!(terminal.failed.len(), 1);
        assert_eq!(scheduler.report().current_in_flight_work, 0);
        assert_eq!(
            scheduler.request_status(request),
            Some(RequestStatus::Failed)
        );
    }

    #[test]
    fn distributed_cancellation_and_deadline_use_the_same_core_lifecycle() {
        let transport =
            ScriptedTransport::new(2, vec![GatherStep::default(), GatherStep::default()]);
        let mut cancelled = scheduler();
        let request = RequestId::new(52);
        cancelled.register(request, State::default()).unwrap();
        cancelled.enqueue(request, 1).unwrap();
        cancelled
            .cancel_distributed(0xCC, request, &transport, consensus_wait(), Instant::now())
            .unwrap();
        assert_eq!(
            cancelled.request_status(request),
            Some(RequestStatus::Cancelled)
        );

        let transport = ScriptedTransport::new(
            2,
            vec![
                GatherStep::default(),
                GatherStep::default(),
                GatherStep::default(),
            ],
        );
        let mut scheduler = scheduler();
        let request = RequestId::new(53);
        scheduler.register(request, State::default()).unwrap();
        scheduler
            .enqueue_with_deadline(
                request,
                1,
                Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
            )
            .unwrap();
        scheduler
            .run_distributed_turn(
                0xDD,
                &transport,
                consensus_wait(),
                Instant::now(),
                |_, _, _| {
                    Ok::<_, Infallible>(Output {
                        complete: Rc::new(Cell::new(true)),
                        fail: false,
                    })
                },
            )
            .unwrap();
        assert_eq!(
            scheduler.request_status(request),
            Some(RequestStatus::DeadlineExceeded)
        );
    }

    #[test]
    fn bounded_distributed_cancellation_rolls_back_failed_phases_and_fences_retry() {
        for steps in [
            vec![
                GatherStep {
                    replacements: vec![(1, 6, 0)],
                },
                GatherStep::default(),
            ],
            vec![
                GatherStep::default(),
                GatherStep {
                    replacements: vec![(1, 6, 0)],
                },
                GatherStep::default(),
            ],
        ] {
            let transport = ScriptedTransport::new(2, steps);
            let mut scheduler = scheduler();
            let request = RequestId::new(53);
            scheduler.register(request, State::default()).unwrap();
            scheduler.enqueue(request, 1).unwrap();

            assert!(scheduler
                .cancel_distributed(0xCD, request, &transport, consensus_wait(), Instant::now(),)
                .is_err());
            assert!(scheduler.request_state(request).is_some());
            assert_eq!(
                scheduler.request_status(request),
                Some(RequestStatus::Active)
            );
            let remaining = transport.steps.borrow().len();
            assert!(scheduler
                .cancel_distributed(0xCD, request, &transport, consensus_wait(), Instant::now(),)
                .is_err());
            assert_eq!(transport.steps.borrow().len(), remaining);
            assert!(scheduler.request_state(request).is_some());
        }
    }

    #[test]
    fn bounded_distributed_deadline_uses_remote_status_and_fences_failed_phases() {
        let remote_expiry = GatherStep {
            replacements: vec![(1, 6, 1)],
        };
        let transport = ScriptedTransport::new(
            2,
            vec![
                remote_expiry,
                GatherStep::default(),
                GatherStep::default(),
                GatherStep::default(),
                GatherStep::default(),
            ],
        );
        let mut successful_scheduler = scheduler();
        let request = RequestId::new(54);
        successful_scheduler
            .register(request, State::default())
            .unwrap();
        successful_scheduler.enqueue(request, 1).unwrap();
        successful_scheduler
            .run_distributed_turn(
                0xCE,
                &transport,
                consensus_wait(),
                Instant::now(),
                |_, _, _| {
                    Ok::<_, Infallible>(Output {
                        complete: Rc::new(Cell::new(true)),
                        fail: false,
                    })
                },
            )
            .unwrap();
        assert_eq!(
            successful_scheduler.request_status(request),
            Some(RequestStatus::DeadlineExceeded)
        );

        for steps in [
            vec![
                GatherStep {
                    replacements: vec![(1, 6, 1)],
                },
                GatherStep {
                    replacements: vec![(1, 6, 0)],
                },
            ],
            vec![
                GatherStep {
                    replacements: vec![(1, 6, 1)],
                },
                GatherStep::default(),
                GatherStep {
                    replacements: vec![(1, 6, 0)],
                },
            ],
        ] {
            let transport = ScriptedTransport::new(2, steps);
            let mut scheduler = scheduler();
            let request = RequestId::new(55);
            scheduler.register(request, State::default()).unwrap();
            scheduler.enqueue(request, 1).unwrap();
            assert!(scheduler
                .run_distributed_turn(
                    0xCF,
                    &transport,
                    consensus_wait(),
                    Instant::now(),
                    |_, _, _| {
                        Ok::<_, Infallible>(Output {
                            complete: Rc::new(Cell::new(true)),
                            fail: false,
                        })
                    },
                )
                .is_err());
            assert_eq!(
                scheduler.request_status(request),
                Some(RequestStatus::Active)
            );
            let remaining = transport.steps.borrow().len();
            assert!(scheduler
                .run_distributed_turn(
                    0xCF,
                    &transport,
                    consensus_wait(),
                    Instant::now(),
                    |_, _, _| {
                        Ok::<_, Infallible>(Output {
                            complete: Rc::new(Cell::new(true)),
                            fail: false,
                        })
                    },
                )
                .is_err());
            assert_eq!(transport.steps.borrow().len(), remaining);
        }
    }

    #[test]
    fn production_telemetry_round_trips_without_backend_types() {
        let mut scheduler = scheduler();
        let request = RequestId::new(40);
        scheduler.register(request, State::default()).unwrap();
        scheduler.enqueue(request, 1).unwrap();
        scheduler.prepare_bounded(1, Instant::now()).unwrap();
        scheduler
            .submit_prepared(Instant::now(), |_, _, _| {
                Ok::<_, Infallible>(Output {
                    complete: Rc::new(Cell::new(false)),
                    fail: false,
                })
            })
            .unwrap();

        let report = scheduler.report();
        let report_json = serde_json::to_string(&report).unwrap();
        assert_eq!(
            serde_json::from_str::<SchedulerReport>(&report_json).unwrap(),
            report
        );

        let capabilities = scheduler.capabilities();
        assert_eq!(capabilities.observed_backends, ["mock"]);
        let capabilities_json = serde_json::to_string(&capabilities).unwrap();
        assert_eq!(
            serde_json::from_str::<SchedulerCapabilities>(&capabilities_json).unwrap(),
            capabilities
        );
    }
}

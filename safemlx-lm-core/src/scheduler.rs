//! Fair transactional scheduling independent of event and tensor runtimes.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

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
}

/// Exact backend completion retaining submitted resources.
pub trait TransitionOutput {
    /// Completion observation error.
    type Error: std::error::Error;
    /// Nonblocking exact completion observation.
    fn is_complete(&self) -> Result<bool, Self::Error>;
    /// Explicitly retained resource count.
    fn retained_resources(&self) -> usize;
}

struct Request<W, S> {
    state: S,
    next: u64,
    pending: VecDeque<(WorkId, W)>,
}
struct Prepared<W, B> {
    id: WorkId,
    work: W,
    branch: B,
}
struct Submitted<W, B, O> {
    id: WorkId,
    work: W,
    branch: B,
    output: O,
    abandoned: bool,
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

/// Snapshot of scheduler occupancy and totals.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct SchedulerReport {
    /// Active requests.
    pub active_requests: usize,
    /// Queued work.
    pub queued_work: usize,
    /// Prepared work.
    pub prepared_work: usize,
    /// Submitted work, including abandoned work.
    pub in_flight_work: usize,
    /// Abandoned retained resources.
    pub abandoned_retained_resources: usize,
    /// Total committed transitions.
    pub committed_work: u64,
    /// Total failed transitions.
    pub failed_work: u64,
}

/// Fair scheduler whose state transitions are independent of backend objects.
pub struct Scheduler<W, S: SemanticStateTransaction, O: TransitionOutput> {
    limits: SchedulerLimits,
    requests: BTreeMap<RequestId, Request<W, S>>,
    ready: VecDeque<RequestId>,
    prepared: VecDeque<Prepared<W, S::Branch>>,
    submitted: Vec<Submitted<W, S::Branch, O>>,
    lifecycle: BTreeMap<WorkId, WorkLifecycle>,
    terminal: BTreeMap<RequestId, RequestStatus>,
    committed: u64,
    failed: u64,
}

impl<W, S: SemanticStateTransaction, O: TransitionOutput> Scheduler<W, S, O> {
    /// Creates an empty scheduler.
    pub fn new(limits: SchedulerLimits) -> Self {
        Self {
            limits,
            requests: BTreeMap::new(),
            ready: VecDeque::new(),
            prepared: VecDeque::new(),
            submitted: vec![],
            lifecycle: BTreeMap::new(),
            terminal: BTreeMap::new(),
            committed: 0,
            failed: 0,
        }
    }
    /// Registers canonical request state.
    pub fn register(&mut self, id: RequestId, state: S) -> Result<(), SchedulerError> {
        if self.requests.contains_key(&id) || self.terminal.contains_key(&id) {
            return Err(SchedulerError::DuplicateRequest(id));
        }
        if self.requests.len() >= self.limits.max_active_requests {
            return Err(SchedulerError::Capacity);
        }
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
    /// Enqueues work in request order.
    pub fn enqueue(&mut self, request: RequestId, work: W) -> Result<WorkId, SchedulerError> {
        if self.nonterminal_work() >= self.limits.max_queued_work {
            return Err(SchedulerError::Capacity);
        }
        let entry = self
            .requests
            .get_mut(&request)
            .ok_or(SchedulerError::UnknownRequest(request))?;
        let id = WorkId {
            request,
            sequence: entry.next,
        };
        entry.next += 1;
        let was_empty = entry.pending.is_empty();
        entry.pending.push_back((id, work));
        if was_empty {
            self.ready.push_back(request);
        }
        self.lifecycle.insert(id, WorkLifecycle::Queued);
        Ok(id)
    }
    /// Prepares up to `limit` branches in round-robin request order.
    pub fn prepare(&mut self, limit: usize) -> Result<usize, SchedulerError> {
        let mut count = 0;
        while count < limit {
            let Some(request_id) = self.ready.pop_front() else {
                break;
            };
            let request = self
                .requests
                .get_mut(&request_id)
                .expect("ready request exists");
            let Some((id, work)) = request.pending.pop_front() else {
                continue;
            };
            if !request.pending.is_empty() {
                self.ready.push_back(request_id);
            }
            match request.state.branch() {
                Ok(branch) => {
                    self.prepared.push_back(Prepared { id, work, branch });
                    self.lifecycle.insert(id, WorkLifecycle::Prepared);
                    count += 1;
                }
                Err(error) => {
                    self.lifecycle.insert(id, WorkLifecycle::Failed);
                    self.failed += 1;
                    self.terminal.insert(request_id, RequestStatus::Failed);
                    return Err(SchedulerError::State(error.to_string()));
                }
            }
        }
        Ok(count)
    }
    /// Submits prepared work through the injected backend adapter.
    pub fn submit(
        &mut self,
        mut execute: impl FnMut(WorkId, &W, &mut S::Branch) -> Result<O, SchedulerError>,
    ) -> Result<usize, SchedulerError> {
        let capacity = self
            .limits
            .max_in_flight_global
            .saturating_sub(self.submitted.len())
            .min(self.limits.max_new_submissions_per_turn);
        let mut count = 0;
        while count < capacity {
            let Some(mut work) = self.prepared.pop_front() else {
                break;
            };
            match execute(work.id, &work.work, &mut work.branch) {
                Ok(output) => {
                    self.lifecycle.insert(work.id, WorkLifecycle::Submitted);
                    self.submitted.push(Submitted {
                        id: work.id,
                        work: work.work,
                        branch: work.branch,
                        output,
                        abandoned: false,
                    });
                    count += 1;
                }
                Err(error) => {
                    let _ = S::discard_branch(work.branch);
                    self.lifecycle.insert(work.id, WorkLifecycle::Failed);
                    self.failed += 1;
                    self.terminal
                        .insert(work.id.request(), RequestStatus::Failed);
                    return Err(error);
                }
            }
        }
        Ok(count)
    }
    /// Polls exact completions and commits only non-abandoned branches.
    pub fn poll(&mut self) -> SchedulerProgress<W, O> {
        let mut progress = SchedulerProgress::default();
        let mut retained = Vec::new();
        for work in std::mem::take(&mut self.submitted) {
            match work.output.is_complete() {
                Ok(false) => retained.push(work),
                Ok(true) if work.abandoned => {
                    let _ = S::discard_branch(work.branch);
                }
                Ok(true) => {
                    self.lifecycle.insert(work.id, WorkLifecycle::Completing);
                    let state = &mut self
                        .requests
                        .get_mut(&work.id.request())
                        .expect("active request")
                        .state;
                    match state.commit_branch(work.branch) {
                        Ok(()) => {
                            self.lifecycle.insert(work.id, WorkLifecycle::Committed);
                            self.committed += 1;
                            progress.committed.push((work.id, work.work, work.output));
                        }
                        Err(error) => {
                            self.lifecycle.insert(work.id, WorkLifecycle::Failed);
                            self.failed += 1;
                            self.terminal
                                .insert(work.id.request(), RequestStatus::Failed);
                            progress
                                .failed
                                .push((work.id, SchedulerError::State(error.to_string())));
                        }
                    }
                }
                Err(error) => {
                    let id = work.id;
                    let _ = S::discard_branch(work.branch);
                    self.lifecycle.insert(id, WorkLifecycle::Failed);
                    self.failed += 1;
                    self.terminal.insert(id.request(), RequestStatus::Failed);
                    progress
                        .failed
                        .push((id, SchedulerError::Completion(error.to_string())));
                }
            }
        }
        self.submitted = retained;
        progress
    }
    /// Cancels unpublished work and marks submitted work abandoned until exact completion.
    pub fn cancel(
        &mut self,
        request: RequestId,
        cause: CancellationCause,
    ) -> Result<(), SchedulerError> {
        let entry = self
            .requests
            .get_mut(&request)
            .ok_or(SchedulerError::UnknownRequest(request))?;
        for (id, _) in entry.pending.drain(..) {
            self.lifecycle.insert(id, WorkLifecycle::Abandoned);
        }
        self.ready.retain(|id| *id != request);
        let mut retained = VecDeque::new();
        for work in self.prepared.drain(..) {
            if work.id.request() == request {
                let _ = S::discard_branch(work.branch);
                self.lifecycle.insert(work.id, WorkLifecycle::Abandoned);
            } else {
                retained.push_back(work);
            }
        }
        self.prepared = retained;
        for work in &mut self.submitted {
            if work.id.request() == request {
                work.abandoned = true;
                self.lifecycle.insert(work.id, WorkLifecycle::Abandoned);
            }
        }
        self.terminal.insert(
            request,
            match cause {
                CancellationCause::Explicit => RequestStatus::Cancelled,
                CancellationCause::Deadline => RequestStatus::DeadlineExceeded,
            },
        );
        Ok(())
    }
    /// Returns a lifecycle state.
    pub fn work_lifecycle(&self, id: WorkId) -> Option<WorkLifecycle> {
        self.lifecycle.get(&id).copied()
    }
    /// Returns request state.
    pub fn request_state(&self, id: RequestId) -> Option<&S> {
        self.requests.get(&id).map(|entry| &entry.state)
    }
    /// Returns active or terminal request status.
    pub fn request_status(&self, id: RequestId) -> Option<RequestStatus> {
        if self.requests.contains_key(&id) && !self.terminal.contains_key(&id) {
            Some(RequestStatus::Active)
        } else {
            self.terminal.get(&id).copied()
        }
    }
    /// Returns scheduler accounting.
    pub fn report(&self) -> SchedulerReport {
        SchedulerReport {
            active_requests: self.requests.len(),
            queued_work: self.requests.values().map(|r| r.pending.len()).sum(),
            prepared_work: self.prepared.len(),
            in_flight_work: self.submitted.len(),
            abandoned_retained_resources: self
                .submitted
                .iter()
                .filter(|w| w.abandoned)
                .map(|w| w.output.retained_resources())
                .sum(),
            committed_work: self.committed,
            failed_work: self.failed,
        }
    }
    fn nonterminal_work(&self) -> usize {
        self.requests
            .values()
            .map(|r| r.pending.len())
            .sum::<usize>()
            + self.prepared.len()
            + self.submitted.len()
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
    /// Capacity exhausted.
    #[error("scheduler capacity exhausted")]
    Capacity,
    /// Semantic state operation failed.
    #[error("semantic state transaction failed: {0}")]
    State(String),
    /// Backend submission failed.
    #[error("backend submission failed: {0}")]
    Submission(String),
    /// Exact completion observation failed.
    #[error("exact completion observation failed: {0}")]
    Completion(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, convert::Infallible, rc::Rc};
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
    }
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
        fn retained_resources(&self) -> usize {
            2
        }
    }
    fn scheduler() -> Scheduler<u32, State, Output> {
        Scheduler::new(SchedulerLimits::default())
    }

    #[test]
    fn queued_prepared_submitted_committed_exactly() {
        let done = Rc::new(Cell::new(false));
        let mut scheduler = scheduler();
        let request = RequestId::new(1);
        scheduler.register(request, State::default()).unwrap();
        let id = scheduler.enqueue(request, 7).unwrap();
        assert_eq!(scheduler.work_lifecycle(id), Some(WorkLifecycle::Queued));
        scheduler.prepare(1).unwrap();
        assert_eq!(scheduler.work_lifecycle(id), Some(WorkLifecycle::Prepared));
        scheduler
            .submit(|_, _, _| {
                Ok(Output {
                    complete: done.clone(),
                    fail: false,
                })
            })
            .unwrap();
        assert!(scheduler.poll().committed.is_empty());
        done.set(true);
        assert_eq!(scheduler.poll().committed.len(), 1);
        assert_eq!(scheduler.request_state(request).unwrap().0, 1);
    }
    #[test]
    fn cancellation_retains_submitted_resources_until_completion() {
        let done = Rc::new(Cell::new(false));
        let mut scheduler = scheduler();
        let request = RequestId::new(2);
        scheduler.register(request, State::default()).unwrap();
        let id = scheduler.enqueue(request, 1).unwrap();
        scheduler.prepare(1).unwrap();
        scheduler
            .submit(|_, _, _| {
                Ok(Output {
                    complete: done.clone(),
                    fail: false,
                })
            })
            .unwrap();
        scheduler
            .cancel(request, CancellationCause::Explicit)
            .unwrap();
        assert_eq!(scheduler.work_lifecycle(id), Some(WorkLifecycle::Abandoned));
        assert_eq!(scheduler.report().abandoned_retained_resources, 2);
        assert_eq!(
            scheduler.request_status(request),
            Some(RequestStatus::Cancelled)
        );
        done.set(true);
        scheduler.poll();
        assert_eq!(scheduler.report().in_flight_work, 0);
        assert_eq!(scheduler.request_state(request).unwrap().0, 0);
    }
    #[test]
    fn exact_completion_failure_does_not_commit() {
        let mut scheduler = scheduler();
        let request = RequestId::new(3);
        scheduler.register(request, State::default()).unwrap();
        let id = scheduler.enqueue(request, 1).unwrap();
        scheduler.prepare(1).unwrap();
        scheduler
            .submit(|_, _, _| {
                Ok(Output {
                    complete: Rc::new(Cell::new(true)),
                    fail: true,
                })
            })
            .unwrap();
        assert_eq!(scheduler.poll().failed.len(), 1);
        assert_eq!(scheduler.work_lifecycle(id), Some(WorkLifecycle::Failed));
        assert_eq!(scheduler.request_state(request).unwrap().0, 0);
    }

    #[test]
    fn submission_failure_marks_work_and_request_failed() {
        let mut scheduler = scheduler();
        let request = RequestId::new(4);
        scheduler.register(request, State::default()).unwrap();
        let id = scheduler.enqueue(request, 1).unwrap();
        scheduler.prepare(1).unwrap();
        let error = scheduler
            .submit(|_, _, _| Err(SchedulerError::Submission("mock submit".into())))
            .unwrap_err();
        assert!(matches!(error, SchedulerError::Submission(_)));
        assert_eq!(scheduler.work_lifecycle(id), Some(WorkLifecycle::Failed));
        assert_eq!(
            scheduler.request_status(request),
            Some(RequestStatus::Failed)
        );
        assert_eq!(scheduler.request_state(request).unwrap().0, 0);
    }
}

//! Architecture-neutral local and distributed request scheduling.
//!
//! The runtime owns request/work identity, bounded fair queues, request state,
//! lifecycle transitions, exact cross-rank work consensus, failure poisoning,
//! and telemetry. Programs such as decoder inference and realtime temporal /
//! depth execution supply only their state, work payload, descriptor encoding,
//! and executor closure.

use std::collections::{BTreeMap, VecDeque};

use safemlx::{
    distributed::{self, Group},
    transforms::eval,
    Array, Stream,
};

use crate::error::Error;

/// Stable caller-assigned identity for one scheduler-owned request.
#[derive(Debug, Clone, Copy, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct RequestId(u64);

impl RequestId {
    /// Creates an identity. Zero is valid; uniqueness is scheduler-local.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the caller-assigned numeric identity.
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// Scheduler-assigned identity for one ordered request-state transition.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct WorkId {
    request: RequestId,
    sequence: u64,
}

impl WorkId {
    /// Returns the request whose state this work updates.
    pub const fn request(self) -> RequestId {
        self.request
    }

    /// Returns the zero-based transition number within the request.
    pub const fn sequence(self) -> u64 {
        self.sequence
    }
}

/// Exact cross-rank descriptor for a program-specific work payload.
///
/// Implementations append deterministic `u32` words for every semantic field
/// that must agree before distributed execution. Request identity and work
/// sequence are encoded by [`FairScheduler`] and must not be duplicated here.
pub trait WorkDescriptor {
    /// Appends the exact semantic descriptor in stable wire order.
    fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Error>;
}

/// Capacity controls shared by every scheduler program.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SchedulerLimits {
    /// Maximum number of simultaneously state-owning requests.
    pub max_active_requests: usize,
    /// Maximum submitted but not yet executed work items.
    pub max_queued_work: usize,
}

impl SchedulerLimits {
    /// Creates positive request and queue bounds.
    pub fn new(max_active_requests: usize, max_queued_work: usize) -> Result<Self, Error> {
        if max_active_requests == 0 || max_queued_work == 0 {
            return Err(Error::Parallel(format!(
                "distributed scheduler capacities must be positive, got {max_active_requests} active requests and {max_queued_work} queued work items"
            )));
        }
        Ok(Self {
            max_active_requests,
            max_queued_work,
        })
    }
}

impl Default for SchedulerLimits {
    fn default() -> Self {
        Self {
            max_active_requests: 64,
            max_queued_work: 256,
        }
    }
}

/// Observable lifecycle state for a scheduler-owned request.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum RequestStatus {
    /// The request owns state and may accept work.
    Active,
    /// A normal terminal condition released the state.
    Finished,
    /// Explicit cancellation released the state and discarded queued work.
    Cancelled,
    /// A distributed or execution error invalidated state ordering.
    Failed,
}

/// Snapshot of generic queue, lifecycle, and throughput counters.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SchedulerReport {
    /// Requests that currently own state.
    pub active_requests: usize,
    /// Submitted work waiting for a drain cycle.
    pub queued_work: usize,
    /// Largest observed queue occupancy.
    pub peak_queued_work: usize,
    /// Total accepted work items.
    pub submitted_work: u64,
    /// Total successfully executed work items.
    pub completed_work: u64,
    /// Work items whose executor returned an error.
    pub failed_work: u64,
    /// Queued work discarded by finish, cancellation, or failure.
    pub discarded_work: u64,
    /// Requests ended normally.
    pub finished_requests: u64,
    /// Requests cancelled explicitly.
    pub cancelled_requests: u64,
    /// Successful queue drain cycles.
    pub drain_cycles: u64,
    /// Whether a failure made request state unsafe to resume.
    pub poisoned: bool,
}

/// One successfully completed scheduler transition.
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

    /// Returns the program-specific execution output.
    pub const fn output(&self) -> &O {
        &self.output
    }

    /// Consumes the completion into its identity, work, and output.
    pub fn into_parts(self) -> (WorkId, W, O) {
        (self.id, self.work, self.output)
    }
}

#[derive(Debug)]
struct RequestEntry<W, S> {
    state: S,
    pending: VecDeque<QueuedWork<W>>,
    next_sequence: u64,
}

#[derive(Debug, Clone)]
struct QueuedWork<W> {
    id: WorkId,
    work: W,
}

/// Bounded fair scheduler for program-defined request state and work.
///
/// Work is drained in stable round-robin request order while preserving exact
/// submission order within each request. Local programs use
/// [`Self::drain_local`]. Distributed programs use [`Self::drain_distributed`]
/// with a versioned protocol identity; exact descriptors are collectively
/// compared before the executor can issue point-to-point operations.
#[derive(Debug)]
pub struct FairScheduler<W, S> {
    limits: SchedulerLimits,
    requests: BTreeMap<RequestId, RequestEntry<W, S>>,
    terminal: BTreeMap<RequestId, RequestStatus>,
    ready: VecDeque<RequestId>,
    queued_work: usize,
    peak_queued_work: usize,
    submitted_work: u64,
    completed_work: u64,
    failed_work: u64,
    discarded_work: u64,
    finished_requests: u64,
    cancelled_requests: u64,
    drain_cycles: u64,
    poisoned: Option<String>,
}

impl<W, S> FairScheduler<W, S> {
    /// Creates an empty scheduler under validated limits.
    pub fn new(limits: SchedulerLimits) -> Result<Self, Error> {
        let limits = SchedulerLimits::new(limits.max_active_requests, limits.max_queued_work)?;
        Ok(Self {
            limits,
            requests: BTreeMap::new(),
            terminal: BTreeMap::new(),
            ready: VecDeque::new(),
            queued_work: 0,
            peak_queued_work: 0,
            submitted_work: 0,
            completed_work: 0,
            failed_work: 0,
            discarded_work: 0,
            finished_requests: 0,
            cancelled_requests: 0,
            drain_cycles: 0,
            poisoned: None,
        })
    }

    /// Registers a unique request and transfers its state to the scheduler.
    pub fn register(&mut self, request: RequestId, state: S) -> Result<(), Error> {
        self.validate_registration(request)?;
        self.insert_request(request, state);
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
                "distributed scheduler active-request capacity {} is exhausted",
                self.limits.max_active_requests
            )));
        }
        Ok(())
    }

    fn insert_request(&mut self, request: RequestId, state: S) {
        self.requests.insert(
            request,
            RequestEntry {
                state,
                pending: VecDeque::new(),
                next_sequence: 0,
            },
        );
    }

    /// Returns immutable program state for an active request.
    pub fn request_state(&self, request: RequestId) -> Option<&S> {
        self.requests.get(&request).map(|entry| &entry.state)
    }

    /// Returns mutable program state for an active request.
    ///
    /// Programs use this to validate and record submission-time invariants.
    /// Execution-time mutation should occur through a drain method.
    pub fn request_state_mut(&mut self, request: RequestId) -> Result<&mut S, Error> {
        self.ensure_ready()?;
        self.requests
            .get_mut(&request)
            .map(|entry| &mut entry.state)
            .ok_or_else(|| Error::Parallel(format!("request {} is not active", request.value())))
    }

    /// Enqueues one state transition and assigns its per-request sequence.
    pub fn enqueue(&mut self, request: RequestId, work: W) -> Result<WorkId, Error> {
        Ok(self
            .enqueue_batch(request, vec![work])?
            .pop()
            .expect("one submitted work item"))
    }

    /// Atomically enqueues an ordered batch of transitions for one request.
    ///
    /// Capacity and sequence space are validated before any work becomes
    /// visible. An empty batch is a no-op but still requires an active request.
    pub fn enqueue_batch(
        &mut self,
        request: RequestId,
        work: Vec<W>,
    ) -> Result<Vec<WorkId>, Error> {
        self.ensure_ready()?;
        let requested = work.len();
        let queued_after = self.queued_work.checked_add(requested).ok_or_else(|| {
            Error::Parallel("distributed scheduler queue occupancy overflow".into())
        })?;
        if queued_after > self.limits.max_queued_work {
            return Err(Error::Parallel(format!(
                "distributed scheduler queue capacity {} cannot accept {requested} work items with {} already queued",
                self.limits.max_queued_work, self.queued_work
            )));
        }
        let entry = self
            .requests
            .get_mut(&request)
            .ok_or_else(|| Error::Parallel(format!("request {} is not active", request.value())))?;
        let requested_u64 = u64::try_from(requested)
            .map_err(|_| Error::Parallel("distributed work batch length exceeds u64".into()))?;
        let next_sequence = entry
            .next_sequence
            .checked_add(requested_u64)
            .ok_or_else(|| {
                Error::Parallel(format!(
                    "request {} exhausted its work sequence space",
                    request.value()
                ))
            })?;
        let was_empty = entry.pending.is_empty();
        let mut ids = Vec::with_capacity(requested);
        for (offset, work) in work.into_iter().enumerate() {
            let id = WorkId {
                request,
                sequence: entry.next_sequence + offset as u64,
            };
            entry.pending.push_back(QueuedWork { id, work });
            ids.push(id);
        }
        entry.next_sequence = next_sequence;
        if was_empty && requested != 0 {
            self.ready.push_back(request);
        }
        self.queued_work = queued_after;
        self.peak_queued_work = self.peak_queued_work.max(self.queued_work);
        self.submitted_work = self.submitted_work.saturating_add(requested_u64);
        Ok(ids)
    }

    /// Drains the current fair queue without distributed consensus.
    ///
    /// This is the canonical path for a program whose work executes in one
    /// process. Descriptor construction still runs before execution so invalid
    /// work poisons request state under the same fail-closed contract as a
    /// distributed drain.
    pub fn drain_local<O>(
        &mut self,
        execute: impl FnMut(WorkId, &W, &mut S) -> Result<O, Error>,
    ) -> Result<Vec<CompletedWork<W, O>>, Error>
    where
        W: WorkDescriptor,
    {
        self.drain_local_bounded(usize::MAX, execute)
    }

    /// Drains at most `max_work` fair-ordered transitions locally.
    ///
    /// This provides cooperative cancellation/deadline boundaries for realtime
    /// programs without changing queue order or request-state ownership. Only
    /// work inside the selected fair prefix has its descriptor constructed;
    /// later queued work is not planned until a subsequent drain.
    pub fn drain_local_bounded<O>(
        &mut self,
        max_work: usize,
        execute: impl FnMut(WorkId, &W, &mut S) -> Result<O, Error>,
    ) -> Result<Vec<CompletedWork<W, O>>, Error>
    where
        W: WorkDescriptor,
    {
        self.ensure_ready()?;
        if max_work == 0 {
            return Err(Error::Parallel(
                "local scheduler drain bound must be positive".into(),
            ));
        }
        if self.queued_work == 0 {
            return Ok(Vec::new());
        }
        let plan = match self.planned_work(max_work) {
            Ok(plan) => plan,
            Err(error) => {
                self.poison(error.to_string());
                return Err(error);
            }
        };
        self.execute_plan(plan, execute)
    }

    /// Drains the current fair queue after exact cross-rank descriptor consensus.
    ///
    /// `protocol` is a stable program-defined wire identity. Changing descriptor
    /// semantics requires changing this value rather than accepting two ranks
    /// that interpret identical words differently.
    pub fn drain_distributed<O>(
        &mut self,
        protocol: u64,
        group: &Group,
        stream: &Stream,
        execute: impl FnMut(WorkId, &W, &mut S) -> Result<O, Error>,
    ) -> Result<Vec<CompletedWork<W, O>>, Error>
    where
        W: WorkDescriptor,
    {
        self.ensure_ready()?;
        if self.queued_work == 0 {
            return Ok(Vec::new());
        }
        let plan = match self.planned_work(usize::MAX) {
            Ok(plan) => plan,
            Err(error) => {
                self.poison(error.to_string());
                return Err(error);
            }
        };
        if let Err(error) =
            validate_schedule_consensus(&plan, self.drain_cycles, protocol, group, stream)
        {
            self.poison(error.to_string());
            return Err(error);
        }

        self.execute_plan(plan, execute)
    }

    fn execute_plan<O>(
        &mut self,
        plan: Vec<PlannedWork>,
        mut execute: impl FnMut(WorkId, &W, &mut S) -> Result<O, Error>,
    ) -> Result<Vec<CompletedWork<W, O>>, Error> {
        let mut completed = Vec::with_capacity(plan.len());
        for expected in plan {
            let Some(queued) = self.pop_next() else {
                let error =
                    Error::Parallel("distributed scheduler queue changed during drain".into());
                self.poison(error.to_string());
                return Err(error);
            };
            if queued.id != expected.id {
                let error = Error::Parallel(format!(
                    "distributed scheduler produced work {:?}, expected {:?}",
                    queued.id, expected.id
                ));
                self.poison(error.to_string());
                return Err(error);
            }
            let result = {
                let state = &mut self
                    .requests
                    .get_mut(&queued.id.request)
                    .expect("ready request owns state")
                    .state;
                execute(queued.id, &queued.work, state)
            };
            let output = match result {
                Ok(output) => output,
                Err(error) => {
                    self.failed_work = self.failed_work.saturating_add(1);
                    self.poison(error.to_string());
                    return Err(error);
                }
            };
            self.completed_work = self.completed_work.saturating_add(1);
            completed.push(CompletedWork {
                id: queued.id,
                work: queued.work,
                output,
            });
        }
        self.drain_cycles = self.drain_cycles.saturating_add(1);
        Ok(completed)
    }

    /// Marks a request complete and releases its state.
    pub fn finish(&mut self, request: RequestId) -> Result<(), Error> {
        self.terminate(request, RequestStatus::Finished)?;
        self.finished_requests = self.finished_requests.saturating_add(1);
        Ok(())
    }

    /// Cancels a request, releases its state, and discards queued work.
    pub fn cancel(&mut self, request: RequestId) -> Result<(), Error> {
        self.terminate(request, RequestStatus::Cancelled)?;
        self.cancelled_requests = self.cancelled_requests.saturating_add(1);
        Ok(())
    }

    /// Releases an idle active request and returns its program state.
    pub fn release(&mut self, request: RequestId) -> Result<S, Error> {
        self.ensure_ready()?;
        let entry = self
            .requests
            .get(&request)
            .ok_or_else(|| Error::Parallel(format!("request {} is not active", request.value())))?;
        if !entry.pending.is_empty() {
            return Err(Error::Parallel(format!(
                "request {} has {} queued work items",
                request.value(),
                entry.pending.len()
            )));
        }
        Ok(self
            .requests
            .remove(&request)
            .expect("checked active request")
            .state)
    }

    /// Removes a terminal identity so a caller may explicitly reuse it.
    pub fn forget_terminal(&mut self, request: RequestId) -> Result<RequestStatus, Error> {
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

    /// Returns the number of queued transitions for one active request.
    pub fn queued_for_request(&self, request: RequestId) -> usize {
        self.requests
            .get(&request)
            .map_or(0, |entry| entry.pending.len())
    }

    /// Returns a current observability snapshot.
    pub fn report(&self) -> SchedulerReport {
        SchedulerReport {
            active_requests: self.requests.len(),
            queued_work: self.queued_work,
            peak_queued_work: self.peak_queued_work,
            submitted_work: self.submitted_work,
            completed_work: self.completed_work,
            failed_work: self.failed_work,
            discarded_work: self.discarded_work,
            finished_requests: self.finished_requests,
            cancelled_requests: self.cancelled_requests,
            drain_cycles: self.drain_cycles,
            poisoned: self.poisoned.is_some(),
        }
    }

    /// Returns the failure that invalidated this scheduler, if any.
    pub fn poison_reason(&self) -> Option<&str> {
        self.poisoned.as_deref()
    }

    fn ensure_ready(&self) -> Result<(), Error> {
        if let Some(reason) = &self.poisoned {
            Err(Error::Parallel(format!(
                "distributed scheduler is poisoned after an earlier failure: {reason}"
            )))
        } else {
            Ok(())
        }
    }

    fn planned_work(&self, max_work: usize) -> Result<Vec<PlannedWork>, Error>
    where
        W: WorkDescriptor,
    {
        let planned_len = self.queued_work.min(max_work);
        let mut ready = self.ready.clone();
        let mut offsets = BTreeMap::<RequestId, usize>::new();
        let mut plan = Vec::with_capacity(planned_len);
        while plan.len() < planned_len {
            let request_id = ready.pop_front().ok_or_else(|| {
                Error::Parallel(format!(
                    "distributed scheduler exhausted its fair queue after planning {} of {planned_len} requested work items",
                    plan.len()
                ))
            })?;
            let request = self.requests.get(&request_id).ok_or_else(|| {
                Error::Parallel(format!(
                    "distributed scheduler ready queue references missing request {}",
                    request_id.value()
                ))
            })?;
            let offset = offsets.entry(request_id).or_default();
            let queued = request.pending.get(*offset).ok_or_else(|| {
                Error::Parallel(format!(
                    "distributed scheduler ready queue references empty request {}",
                    request_id.value()
                ))
            })?;
            let mut descriptor = Vec::new();
            queued.work.encode_descriptor(&mut descriptor)?;
            plan.push(PlannedWork {
                id: queued.id,
                descriptor,
            });
            *offset += 1;
            if *offset < request.pending.len() {
                ready.push_back(request_id);
            }
        }
        if plan.len() != planned_len {
            return Err(Error::Parallel(format!(
                "distributed scheduler planned {} work items but requested {planned_len}",
                plan.len(),
            )));
        }
        Ok(plan)
    }

    fn pop_next(&mut self) -> Option<QueuedWork<W>> {
        let request_id = self.ready.pop_front()?;
        let request = self.requests.get_mut(&request_id)?;
        let work = request.pending.pop_front()?;
        if !request.pending.is_empty() {
            self.ready.push_back(request_id);
        }
        self.queued_work = self.queued_work.saturating_sub(1);
        Some(work)
    }

    fn terminate(&mut self, request: RequestId, status: RequestStatus) -> Result<(), Error> {
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
        let discarded = entry.pending.len();
        self.queued_work = self.queued_work.saturating_sub(discarded);
        self.discarded_work = self.discarded_work.saturating_add(discarded as u64);
        self.ready.retain(|queued| *queued != request);
        self.terminal.insert(request, status);
        Ok(())
    }

    fn poison(&mut self, reason: String) {
        if self.poisoned.is_some() {
            return;
        }
        self.discarded_work = self.discarded_work.saturating_add(self.queued_work as u64);
        self.queued_work = 0;
        self.ready.clear();
        for (request, _) in std::mem::take(&mut self.requests) {
            self.terminal.insert(request, RequestStatus::Failed);
        }
        self.poisoned = Some(reason);
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct PlannedWork {
    id: WorkId,
    descriptor: Vec<u32>,
}

fn validate_schedule_consensus(
    plan: &[PlannedWork],
    drain_cycle: u64,
    protocol: u64,
    group: &Group,
    stream: &Stream,
) -> Result<(), Error> {
    let count = u32::try_from(plan.len())
        .map_err(|_| Error::Parallel("distributed work count exceeds u32".into()))?;
    let mut packed = Vec::new();
    for work in plan {
        push_u64(&mut packed, work.id.request().value());
        push_u64(&mut packed, work.id.sequence());
        packed.push(u32::try_from(work.descriptor.len()).map_err(|_| {
            Error::Parallel("distributed work descriptor length exceeds u32".into())
        })?);
        packed.extend_from_slice(&work.descriptor);
    }
    let packed_len = i32::try_from(packed.len())
        .map_err(|_| Error::Parallel("distributed schedule metadata exceeds i32".into()))?;
    let header = [
        count,
        drain_cycle as u32,
        (drain_cycle >> 32) as u32,
        protocol as u32,
        (protocol >> 32) as u32,
        packed_len as u32,
    ];
    let local_header = Array::from_slice(&header, &[header.len() as i32]);
    let gathered_header =
        distributed::all_gather(&local_header, group, stream).map_err(|error| {
            Error::Parallel(format!(
                "distributed scheduler failed to gather schedule headers: {error}"
            ))
        })?;
    eval([&gathered_header])?;
    stream.synchronize()?;
    let evaluated_header = gathered_header.evaluated()?;
    let gathered_header = evaluated_header.as_slice::<u32>();
    for (rank, candidate) in gathered_header.chunks_exact(header.len()).enumerate() {
        if candidate != header {
            return Err(Error::Parallel(format!(
                "distributed schedule header differs at rank {rank}: {candidate:?} versus {header:?}"
            )));
        }
    }

    let local = Array::from_slice(&packed, &[packed_len]);
    let gathered = distributed::all_gather(&local, group, stream).map_err(|error| {
        Error::Parallel(format!(
            "distributed scheduler failed to gather work descriptors: {error}"
        ))
    })?;
    eval([&gathered])?;
    stream.synchronize()?;
    let evaluated = gathered.evaluated()?;
    let gathered = evaluated.as_slice::<u32>();
    for rank in 0..group.size() {
        let start = rank * packed.len();
        let end = start + packed.len();
        if gathered.get(start..end) != Some(packed.as_slice()) {
            return Err(Error::Parallel(format!(
                "distributed work descriptors differ at rank {rank}"
            )));
        }
    }
    Ok(())
}

fn push_u64(output: &mut Vec<u32>, value: u64) {
    output.extend_from_slice(&[value as u32, (value >> 32) as u32]);
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use super::*;

    #[derive(Debug, Clone)]
    struct TestWork(u32);

    impl WorkDescriptor for TestWork {
        fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Error> {
            output.push(self.0);
            Ok(())
        }
    }

    #[test]
    fn fair_queue_lifecycle_and_backpressure_are_program_independent() {
        let mut scheduler = FairScheduler::new(SchedulerLimits::new(2, 4).unwrap()).unwrap();
        let first = RequestId::new(11);
        let second = RequestId::new(22);
        scheduler.register(first, "first").unwrap();
        scheduler.register(second, "second").unwrap();
        assert!(scheduler.register(RequestId::new(33), "third").is_err());
        scheduler.enqueue(first, TestWork(1)).unwrap();
        scheduler.enqueue(first, TestWork(2)).unwrap();
        scheduler.enqueue(second, TestWork(3)).unwrap();
        scheduler.enqueue(second, TestWork(4)).unwrap();
        assert!(scheduler.enqueue(second, TestWork(5)).is_err());
        assert_eq!(
            scheduler
                .planned_work(usize::MAX)
                .unwrap()
                .iter()
                .map(|work| (work.id.request().value(), work.id.sequence()))
                .collect::<Vec<_>>(),
            vec![(11, 0), (22, 0), (11, 1), (22, 1)]
        );
        scheduler.cancel(first).unwrap();
        scheduler.finish(second).unwrap();
        assert_eq!(
            scheduler.request_status(first),
            Some(RequestStatus::Cancelled)
        );
        assert_eq!(
            scheduler.request_status(second),
            Some(RequestStatus::Finished)
        );
        assert_eq!(
            scheduler.report(),
            SchedulerReport {
                active_requests: 0,
                queued_work: 0,
                peak_queued_work: 4,
                submitted_work: 4,
                completed_work: 0,
                failed_work: 0,
                discarded_work: 4,
                finished_requests: 1,
                cancelled_requests: 1,
                drain_cycles: 0,
                poisoned: false,
            }
        );
    }

    #[test]
    fn state_is_isolated_and_idle_release_is_exact() {
        let mut scheduler =
            FairScheduler::<TestWork, Vec<u32>>::new(SchedulerLimits::new(2, 2).unwrap()).unwrap();
        let request = RequestId::new(7);
        scheduler.register(request, vec![1, 2]).unwrap();
        scheduler.request_state_mut(request).unwrap().push(3);
        assert_eq!(scheduler.request_state(request).unwrap(), &[1, 2, 3]);
        assert_eq!(scheduler.release(request).unwrap(), vec![1, 2, 3]);
        assert_eq!(scheduler.request_status(request), None);
    }

    #[test]
    fn local_drain_uses_the_same_fair_state_machine() {
        let mut scheduler =
            FairScheduler::<TestWork, Vec<u32>>::new(SchedulerLimits::new(2, 4).unwrap()).unwrap();
        let first = RequestId::new(11);
        let second = RequestId::new(22);
        scheduler.register(first, Vec::new()).unwrap();
        scheduler.register(second, Vec::new()).unwrap();
        scheduler.enqueue(first, TestWork(1)).unwrap();
        scheduler.enqueue(first, TestWork(2)).unwrap();
        scheduler.enqueue(second, TestWork(3)).unwrap();
        scheduler.enqueue(second, TestWork(4)).unwrap();

        let completed = scheduler
            .drain_local(|id, work, state| {
                state.push(work.0);
                Ok((id.request().value(), work.0))
            })
            .unwrap();
        assert_eq!(
            completed
                .into_iter()
                .map(CompletedWork::into_parts)
                .map(|(_, _, output)| output)
                .collect::<Vec<_>>(),
            vec![(11, 1), (22, 3), (11, 2), (22, 4)]
        );
        assert_eq!(scheduler.request_state(first).unwrap(), &[1, 2]);
        assert_eq!(scheduler.request_state(second).unwrap(), &[3, 4]);
        assert_eq!(scheduler.report().completed_work, 4);
        assert_eq!(scheduler.report().drain_cycles, 1);
    }

    #[derive(Debug, Clone)]
    struct CountedWork {
        value: u32,
        descriptor_encodes: Arc<AtomicUsize>,
    }

    impl WorkDescriptor for CountedWork {
        fn encode_descriptor(&self, output: &mut Vec<u32>) -> Result<(), Error> {
            self.descriptor_encodes.fetch_add(1, Ordering::Relaxed);
            output.push(self.value);
            Ok(())
        }
    }

    #[test]
    fn bounded_local_drain_encodes_only_the_requested_fair_prefix() {
        let descriptor_encodes = Arc::new(AtomicUsize::new(0));
        let mut scheduler =
            FairScheduler::<CountedWork, ()>::new(SchedulerLimits::new(2, 8).unwrap()).unwrap();
        let first = RequestId::new(11);
        let second = RequestId::new(22);
        scheduler.register(first, ()).unwrap();
        scheduler.register(second, ()).unwrap();
        for value in [1, 2, 3, 4] {
            scheduler
                .enqueue(
                    first,
                    CountedWork {
                        value,
                        descriptor_encodes: Arc::clone(&descriptor_encodes),
                    },
                )
                .unwrap();
        }
        for value in [5, 6, 7, 8] {
            scheduler
                .enqueue(
                    second,
                    CountedWork {
                        value,
                        descriptor_encodes: Arc::clone(&descriptor_encodes),
                    },
                )
                .unwrap();
        }

        let first_prefix = scheduler
            .drain_local_bounded(1, |_, work, _| Ok(work.value))
            .unwrap();
        assert_eq!(descriptor_encodes.load(Ordering::Relaxed), 1);
        assert_eq!(
            first_prefix
                .into_iter()
                .map(CompletedWork::into_parts)
                .map(|(_, _, output)| output)
                .collect::<Vec<_>>(),
            vec![1]
        );
        assert_eq!(scheduler.report().queued_work, 7);

        let second_prefix = scheduler
            .drain_local_bounded(2, |_, work, _| Ok(work.value))
            .unwrap();
        assert_eq!(descriptor_encodes.load(Ordering::Relaxed), 3);
        assert_eq!(
            second_prefix
                .into_iter()
                .map(CompletedWork::into_parts)
                .map(|(_, _, output)| output)
                .collect::<Vec<_>>(),
            vec![5, 2]
        );
        assert_eq!(scheduler.report().queued_work, 5);
    }

    #[test]
    fn local_execution_failure_poison_drops_every_request_state() {
        let mut scheduler =
            FairScheduler::<TestWork, ()>::new(SchedulerLimits::new(2, 2).unwrap()).unwrap();
        let first = RequestId::new(1);
        let second = RequestId::new(2);
        scheduler.register(first, ()).unwrap();
        scheduler.register(second, ()).unwrap();
        scheduler.enqueue(first, TestWork(1)).unwrap();
        scheduler.enqueue(second, TestWork(2)).unwrap();
        let error = scheduler
            .drain_local::<()>(|_, _, _| Err(Error::Parallel("injected failure".into())))
            .unwrap_err();
        assert!(error.to_string().contains("injected failure"));
        assert_eq!(scheduler.request_status(first), Some(RequestStatus::Failed));
        assert_eq!(
            scheduler.request_status(second),
            Some(RequestStatus::Failed)
        );
        assert_eq!(scheduler.report().active_requests, 0);
        assert_eq!(scheduler.report().failed_work, 1);
        assert_eq!(scheduler.report().discarded_work, 1);
        assert!(scheduler.report().poisoned);
        assert!(scheduler.enqueue(first, TestWork(3)).is_err());
    }

    #[test]
    fn batch_submission_is_atomic_under_backpressure() {
        let mut scheduler =
            FairScheduler::<TestWork, ()>::new(SchedulerLimits::new(1, 3).unwrap()).unwrap();
        let request = RequestId::new(9);
        scheduler.register(request, ()).unwrap();
        assert_eq!(
            scheduler.enqueue(request, TestWork(1)).unwrap().sequence(),
            0
        );
        assert!(scheduler
            .enqueue_batch(request, vec![TestWork(2), TestWork(3), TestWork(4)],)
            .is_err());
        assert_eq!(scheduler.queued_for_request(request), 1);
        assert_eq!(scheduler.report().submitted_work, 1);
        assert_eq!(
            scheduler.enqueue(request, TestWork(5)).unwrap().sequence(),
            1
        );
    }
}

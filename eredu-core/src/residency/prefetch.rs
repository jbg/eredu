//! Backend-neutral admission and lifecycle for bounded residency prefetching.

use std::{
    collections::{BTreeMap, VecDeque},
    time::Duration,
};

use serde::{Deserialize, Serialize};

use super::OffloadUnitId;

/// Immutable observations from one bounded background prefetch executor.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct BackgroundPrefetchReport {
    submitted: u64,
    coalesced: u64,
    started: u64,
    completed: u64,
    cancelled: u64,
    failed: u64,
    queue_capacity: usize,
    peak_queue_occupancy: usize,
    backpressure_count: u64,
    backpressure_duration: Duration,
    demand_waits: u64,
    demand_wait_duration: Duration,
    ready_before_demand: u64,
    in_flight_at_demand: u64,
    evicted_before_use: u64,
}

impl BackgroundPrefetchReport {
    /// Accumulates disjoint worker generations with saturating counters and
    /// maximum queue capacity/occupancy. Each generation must be supplied once;
    /// repeated snapshots of the same worker are not disjoint observations.
    /// These observations never grant storage, completion or reuse authority.
    pub fn accumulate(&mut self, other: Self) {
        self.submitted = self.submitted.saturating_add(other.submitted);
        self.coalesced = self.coalesced.saturating_add(other.coalesced);
        self.started = self.started.saturating_add(other.started);
        self.completed = self.completed.saturating_add(other.completed);
        self.cancelled = self.cancelled.saturating_add(other.cancelled);
        self.failed = self.failed.saturating_add(other.failed);
        self.backpressure_count = self.backpressure_count.saturating_add(other.backpressure_count);
        self.backpressure_duration = self.backpressure_duration.saturating_add(other.backpressure_duration);
        self.demand_waits = self.demand_waits.saturating_add(other.demand_waits);
        self.demand_wait_duration = self.demand_wait_duration.saturating_add(other.demand_wait_duration);
        self.ready_before_demand = self.ready_before_demand.saturating_add(other.ready_before_demand);
        self.in_flight_at_demand = self.in_flight_at_demand.saturating_add(other.in_flight_at_demand);
        self.evicted_before_use = self.evicted_before_use.saturating_add(other.evicted_before_use);
        self.queue_capacity = self.queue_capacity.max(other.queue_capacity);
        self.peak_queue_occupancy = self.peak_queue_occupancy.max(other.peak_queue_occupancy);
    }
    /// Requests admitted for background execution.
    pub const fn submitted(self) -> u64 {
        self.submitted
    }

    /// Duplicate or already-resident requests folded into existing work.
    pub const fn coalesced(self) -> u64 {
        self.coalesced
    }

    /// Requests handed to a backend executor.
    pub const fn started(self) -> u64 {
        self.started
    }

    /// Requests published successfully.
    pub const fn completed(self) -> u64 {
        self.completed
    }

    /// Queued or submitted requests discarded by cancellation.
    pub const fn cancelled(self) -> u64 {
        self.cancelled
    }

    /// Backend operations whose failures were retained for demand.
    pub const fn failed(self) -> u64 {
        self.failed
    }

    /// Maximum number of admitted operations awaiting execution.
    pub const fn queue_capacity(self) -> usize {
        self.queue_capacity
    }

    /// Largest observed admitted queue occupancy.
    pub const fn peak_queue_occupancy(self) -> usize {
        self.peak_queue_occupancy
    }

    /// Submissions that encountered full admission capacity.
    pub const fn backpressure_count(self) -> u64 {
        self.backpressure_count
    }

    /// Cumulative wait time for resolved backpressure events.
    pub const fn backpressure_duration(self) -> Duration {
        self.backpressure_duration
    }

    /// Demand acquisitions that waited for admitted or submitted work.
    pub const fn demand_waits(self) -> u64 {
        self.demand_waits
    }

    /// Time spent waiting for demanded work.
    pub const fn demand_wait_duration(self) -> Duration {
        self.demand_wait_duration
    }

    /// Completed prefetches consumed by demand before eviction.
    pub const fn ready_before_demand(self) -> u64 {
        self.ready_before_demand
    }

    /// Prefetches already submitted to the backend when first demanded.
    pub const fn in_flight_at_demand(self) -> u64 {
        self.in_flight_at_demand
    }

    /// Completed prefetches found evicted before first demand.
    pub const fn evicted_before_use(self) -> u64 {
        self.evicted_before_use
    }
}

/// Exact logical operation admitted by [`PrefetchExecutionState`].
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PrefetchWork<K = OffloadUnitId> {
    generation: u64,
    id: K,
    sequence: u64,
}

impl<K> PrefetchWork<K> {
    /// Cancellation generation in which this work was admitted.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Never-reused attempt within this lifecycle owner.
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Logical residency unit to materialize.
    pub const fn id(&self) -> &K {
        &self.id
    }
}

/// Result of attempting to admit one prefetch operation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum PrefetchAdmission<K = OffloadUnitId> {
    /// A new operation was admitted and awaits backend execution.
    Admitted(PrefetchWork<K>),
    /// Existing queued, submitted, completed, or resident state satisfies it.
    Coalesced,
    /// The bounded queue cannot admit another operation yet.
    AtCapacity,
    /// The fixed unit domain or checked attempt issuer rejected this request.
    Rejected(PrefetchStateError<K>),
}

/// State observed when demand first asks for one logical unit.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PrefetchDemandObservation {
    /// The operation is admitted but has not been submitted to the backend.
    Queued,
    /// The backend operation is in flight.
    InFlight,
    /// A successful prefetch is ready to consume.
    Ready,
    /// A backend failure is retained for demand.
    Failed,
    /// No background operation owns the unit.
    Unscheduled,
}

impl PrefetchDemandObservation {
    /// Whether demand must wait for a terminal transition.
    pub const fn is_pending(self) -> bool {
        matches!(self, Self::Queued | Self::InFlight)
    }
}

/// Terminal result consumed by a demand acquisition.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum PrefetchDemandResolution<E> {
    /// Background execution completed and remained available.
    Ready,
    /// Backend execution failed with its structured backend error.
    Failed(E),
    /// Demand must perform or acquire the work directly.
    Unscheduled,
}

/// Publication disposition after an exact backend operation completes.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PrefetchCompletion {
    /// The completed copy became available to demand.
    Published,
    /// The backend failure was retained for demand.
    Failed,
    /// Cancellation made this exact completion stale.
    Discarded,
}

/// Backend-neutral prefetch lifecycle. The default key preserves ordinary named
/// units; [`Self::with_unit_count`] selects a fixed ordinal domain.
///
/// Both representations use the same FIFO, admission, completion, cancellation
/// and telemetry worker. Storage preparation grants no request/native authority.
#[derive(Debug)]
pub struct PrefetchExecutionState<E, K = OffloadUnitId> {
    generation: u64,
    next_sequence: Option<u64>,
    queue_capacity: usize,
    queue: VecDeque<PrefetchWork<K>>,
    slots: Slots<K, E>,
    active: usize,
    report: BackgroundPrefetchReport,
}

#[derive(Debug)]
enum Slot<E> {
    Empty,
    Queued,
    InFlight { generation: u64, sequence: u64 },
    Completed,
    Failed(E),
}
#[derive(Debug)]
enum Slots<K, E> {
    Ordinary(BTreeMap<K, Slot<E>>),
    Fixed(Vec<(K, Slot<E>)>),
}
impl<K: Ord, E> Slots<K, E> {
    fn contains_key(&self, key: &K) -> bool {
        match self {
            Self::Ordinary(_) => true,
            Self::Fixed(slots) => slots.binary_search_by(|(id, _)| id.cmp(key)).is_ok(),
        }
    }
    fn get(&self, key: &K) -> Option<&Slot<E>> {
        match self {
            Self::Ordinary(slots) => slots.get(key),
            Self::Fixed(slots) => slots
                .binary_search_by(|(id, _)| id.cmp(key))
                .ok()
                .map(|i| &slots[i].1),
        }
    }
    fn replace(&mut self, key: K, slot: Slot<E>) -> Option<Slot<E>> {
        match self {
            Self::Ordinary(slots) => {
                if matches!(slot, Slot::Empty) {
                    slots.remove(&key)
                } else {
                    slots.insert(key, slot)
                }
            }
            Self::Fixed(slots) => {
                let index = slots
                    .binary_search_by(|(id, _)| id.cmp(&key))
                    .expect("checked selected unit");
                Some(std::mem::replace(&mut slots[index].1, slot))
            }
        }
    }
    fn first_terminal(&self) -> Option<&K> {
        match self {
            Self::Ordinary(slots) => slots
                .iter()
                .find_map(|(k, s)| matches!(s, Slot::Completed | Slot::Failed(_)).then_some(k)),
            Self::Fixed(slots) => slots
                .iter()
                .find_map(|(k, s)| matches!(s, Slot::Completed | Slot::Failed(_)).then_some(k)),
        }
    }
    fn fixed(&self) -> bool {
        matches!(self, Self::Fixed(_))
    }
}

/// Construction failure preserves every actually reserved fixed-buffer prefix.
/// This is ordinary storage preparation, not an original admission certificate.
#[derive(Debug)]
pub struct PrefetchStorageError<E> {
    cause: PrefetchStorageCause,
    source: Option<std::collections::TryReserveError>,
    slots: Vec<(usize, Slot<E>)>,
    queue: VecDeque<PrefetchWork<usize>>,
}
/// Fixed reason for preparation failure, without allocating a diagnostic.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum PrefetchStorageCause {
    /// The queue capacity must be positive.
    #[error("background prefetch queue capacity must be nonzero")]
    ZeroQueueCapacity,
    /// The actual selected-slot reserve failed.
    #[error("prefetch selected-slot reserve failed")]
    Slots,
    /// The actual FIFO reserve failed.
    #[error("prefetch FIFO reserve failed")]
    Queue,
}
impl<E> PrefetchStorageError<E> {
    /// Fixed constructor failure.
    pub const fn cause(&self) -> PrefetchStorageCause {
        self.cause
    }
    /// Actual retained capacities, selected slots followed by FIFO entries.
    pub fn retained_capacities(&self) -> (usize, usize) {
        (self.slots.capacity(), self.queue.capacity())
    }
}
impl<E> std::fmt::Display for PrefetchStorageError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl<E: std::fmt::Debug> std::error::Error for PrefetchStorageError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|source| source as _)
    }
}

impl<E> PrefetchExecutionState<E> {
    /// Creates ordinary named-unit storage with a finite nonzero FIFO capacity.
    pub fn new(queue_capacity: usize) -> Result<Self, PrefetchStateError> {
        if queue_capacity == 0 {
            return Err(PrefetchStateError::ZeroQueueCapacity);
        }
        Ok(Self::with_storage(
            queue_capacity,
            VecDeque::new(),
            Slots::Ordinary(BTreeMap::new()),
        ))
    }
}
impl<E> PrefetchExecutionState<E, usize> {
    /// Requested payload and fixed constructor controls for this exact selected
    /// slot/FIFO producer. Source IDs, worker synchronization, thread/PAL storage
    /// and host allocator overhead are separate owners.
    pub fn selected_storage_bytes(unit_count: usize, queue_capacity: usize) -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let parts = [
            size_of::<(usize, Slot<E>)>().checked_mul(unit_count)?,
            size_of::<PrefetchWork<usize>>().checked_mul(queue_capacity)?,
            size_of::<Self>(),
            size_of::<PrefetchStorageError<E>>(),
            size_of::<Result<Self, PrefetchStorageError<E>>>(),
            size_of::<std::collections::TryReserveError>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<usize>(),
            size_of::<Option<PrefetchStorageCause>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }

    /// Reserves the actual fixed domain and FIFO before publication. Ordinals
    /// are in `0..unit_count`; only one backend operation may be in flight.
    /// The caller owns the immutable ordinal-to-source mapping. This constructor
    /// does not authorize source work or price worker/thread/synchronization costs.
    pub fn with_unit_count(
        unit_count: usize,
        queue_capacity: usize,
    ) -> Result<Self, PrefetchStorageError<E>> {
        Self::prepare_fixed(unit_count, queue_capacity, None)
    }
    fn prepare_fixed(
        unit_count: usize,
        queue_capacity: usize,
        fail_reserve: Option<PrefetchStorageCause>,
    ) -> Result<Self, PrefetchStorageError<E>> {
        let mut partial = PrefetchStorageError {
            cause: PrefetchStorageCause::ZeroQueueCapacity,
            source: None,
            slots: Vec::new(),
            queue: VecDeque::new(),
        };
        if queue_capacity == 0 {
            return Err(partial);
        }
        partial.cause = PrefetchStorageCause::Slots;
        let count = if fail_reserve == Some(partial.cause) {
            usize::MAX
        } else {
            unit_count
        };
        if let Err(source) = partial.slots.try_reserve_exact(count) {
            partial.source = Some(source);
            return Err(partial);
        }
        partial
            .slots
            .extend((0..unit_count).map(|key| (key, Slot::Empty)));
        partial.cause = PrefetchStorageCause::Queue;
        let count = if fail_reserve == Some(partial.cause) {
            usize::MAX
        } else {
            queue_capacity
        };
        if let Err(source) = partial.queue.try_reserve_exact(count) {
            partial.source = Some(source);
            return Err(partial);
        }
        Ok(Self::with_storage(
            queue_capacity,
            partial.queue,
            Slots::Fixed(partial.slots),
        ))
    }
    /// Actual fixed capacities; no collection can grow during transitions.
    pub fn fixed_capacities(&self) -> (usize, usize) {
        let Slots::Fixed(slots) = &self.slots else {
            unreachable!("ordinal constructor uses fixed slots")
        };
        (slots.capacity(), self.queue.capacity())
    }
}
impl<E, K: Ord + Clone> PrefetchExecutionState<E, K> {
    fn with_storage(
        queue_capacity: usize,
        queue: VecDeque<PrefetchWork<K>>,
        slots: Slots<K, E>,
    ) -> Self {
        Self {
            generation: 0,
            next_sequence: Some(0),
            queue_capacity,
            queue,
            slots,
            active: 0,
            report: BackgroundPrefetchReport {
                queue_capacity,
                ..BackgroundPrefetchReport::default()
            },
        }
    }
    /// Admits or coalesces one request. The owning variant can detach a replaced
    /// failure for destruction after an external lifecycle lock is released.
    pub fn admit(&mut self, id: K, resident: bool) -> PrefetchAdmission<K> {
        self.admit_retaining(id, resident).0
    }
    /// Same transition with any superseded error transferred to the caller.
    pub fn admit_retaining(&mut self, id: K, resident: bool) -> (PrefetchAdmission<K>, Option<E>) {
        if !self.slots.contains_key(&id) {
            return (
                PrefetchAdmission::Rejected(PrefetchStateError::OutsideDomain),
                None,
            );
        }
        if matches!(
            self.slots.get(&id),
            Some(Slot::Queued | Slot::InFlight { .. })
        ) {
            self.report.coalesced = self.report.coalesced.saturating_add(1);
            return (PrefetchAdmission::Coalesced, None);
        }
        // A new attempt refuses atomically if its issuer is exhausted. Resident
        // coalescing and existing-work observations require no new identity.
        if !resident && self.queue.len() < self.queue_capacity && self.next_sequence.is_none() {
            return (
                PrefetchAdmission::Rejected(PrefetchStateError::SequenceExhausted),
                None,
            );
        }
        if matches!(self.slots.get(&id), Some(Slot::Completed)) && !resident {
            self.slots.replace(id.clone(), Slot::Empty);
            self.report.evicted_before_use = self.report.evicted_before_use.saturating_add(1);
        }
        if resident {
            let retired = Self::failure(self.slots.replace(id, Slot::Completed));
            self.report.coalesced = self.report.coalesced.saturating_add(1);
            return (PrefetchAdmission::Coalesced, retired);
        }
        if self.queue.len() == self.queue_capacity {
            return (PrefetchAdmission::AtCapacity, None);
        }
        let Some(sequence) = self.next_sequence else {
            return (
                PrefetchAdmission::Rejected(PrefetchStateError::SequenceExhausted),
                None,
            );
        };
        self.next_sequence = sequence.checked_add(1);
        let work = PrefetchWork {
            generation: self.generation,
            sequence,
            id,
        };
        let retired = Self::failure(self.slots.replace(work.id.clone(), Slot::Queued));
        self.queue.push_back(work.clone());
        self.report.submitted = self.report.submitted.saturating_add(1);
        self.report.peak_queue_occupancy = self.report.peak_queue_occupancy.max(self.queue.len());
        (PrefetchAdmission::Admitted(work), retired)
    }
    fn failure(slot: Option<Slot<E>>) -> Option<E> {
        if let Some(Slot::Failed(error)) = slot {
            Some(error)
        } else {
            None
        }
    }
    /// Rolls back the exact queued attempt after failed worker notification.
    pub fn rollback_admission(
        &mut self,
        work: &PrefetchWork<K>,
    ) -> Result<(), PrefetchStateError<K>> {
        let Some(position) = self.queue.iter().position(|queued| queued == work) else {
            return Err(PrefetchStateError::WorkNotQueued {
                id: work.id.clone(),
                generation: work.generation,
            });
        };
        self.queue.remove(position);
        self.slots.replace(work.id.clone(), Slot::Empty);
        self.report.submitted = self.report.submitted.saturating_sub(1);
        Ok(())
    }
    /// Selects FIFO work; fixed storage supports exactly one worker in flight.
    pub fn begin_next(&mut self) -> Option<PrefetchWork<K>> {
        if self.slots.fixed() && self.active != 0 {
            return None;
        }
        let work = self.queue.pop_front()?;
        self.slots.replace(
            work.id.clone(),
            Slot::InFlight {
                generation: work.generation,
                sequence: work.sequence,
            },
        );
        self.active += 1;
        self.report.started = self.report.started.saturating_add(1);
        Some(work)
    }
    /// Applies an exact completion. Use [`Self::complete_retaining`] when an
    /// external lock must not run destructors of rejected or stale failures.
    pub fn complete(
        &mut self,
        work: PrefetchWork<K>,
        result: Result<(), E>,
    ) -> Result<PrefetchCompletion, PrefetchStateError<K>> {
        self.complete_retaining(work, result)
            .map(|(value, _)| value)
            .map_err(|(error, _)| error)
    }
    /// Returns discarded errors and preserves the supplied result on rejection.
    pub fn complete_retaining(
        &mut self,
        work: PrefetchWork<K>,
        result: Result<(), E>,
    ) -> Result<(PrefetchCompletion, Option<E>), (PrefetchStateError<K>, Result<(), E>)> {
        let Some(Slot::InFlight {
            generation,
            sequence,
        }) = self.slots.get(&work.id)
        else {
            return Err((
                PrefetchStateError::WorkNotInFlight {
                    id: work.id,
                    generation: work.generation,
                },
                result,
            ));
        };
        if *generation != work.generation {
            return Err((
                PrefetchStateError::CompletionGenerationMismatch {
                    id: work.id,
                    expected: *generation,
                    actual: work.generation,
                },
                result,
            ));
        }
        if *sequence != work.sequence {
            return Err((PrefetchStateError::CompletionSequenceMismatch, result));
        }
        self.active -= 1;
        if work.generation != self.generation {
            self.slots.replace(work.id, Slot::Empty);
            self.report.cancelled = self.report.cancelled.saturating_add(1);
            return Ok((PrefetchCompletion::Discarded, result.err()));
        }
        let value = match result {
            Ok(()) => {
                self.slots.replace(work.id, Slot::Completed);
                self.report.completed = self.report.completed.saturating_add(1);
                PrefetchCompletion::Published
            }
            Err(error) => {
                self.slots.replace(work.id, Slot::Failed(error));
                self.report.failed = self.report.failed.saturating_add(1);
                PrefetchCompletion::Failed
            }
        };
        Ok((value, None))
    }
    /// Observes a unit, retaining the ordinary telemetry semantics.
    pub fn observe_demand(&mut self, id: &K) -> PrefetchDemandObservation {
        match self.slots.get(id) {
            Some(Slot::Queued) => PrefetchDemandObservation::Queued,
            Some(Slot::InFlight { .. }) => {
                self.report.in_flight_at_demand = self.report.in_flight_at_demand.saturating_add(1);
                PrefetchDemandObservation::InFlight
            }
            Some(Slot::Failed(_)) => PrefetchDemandObservation::Failed,
            Some(Slot::Completed) => PrefetchDemandObservation::Ready,
            _ => PrefetchDemandObservation::Unscheduled,
        }
    }
    /// Whether the unit is queued or in flight.
    pub fn is_pending(&self, id: &K) -> bool {
        matches!(
            self.slots.get(id),
            Some(Slot::Queued | Slot::InFlight { .. })
        )
    }
    /// Consumes one terminal result.
    pub fn resolve_demand(
        &mut self,
        id: &K,
        waited: Option<Duration>,
    ) -> Result<PrefetchDemandResolution<E>, PrefetchStateError<K>> {
        if self.is_pending(id) {
            return Err(PrefetchStateError::DemandStillPending { id: id.clone() });
        }
        if !self.slots.contains_key(id) {
            return Err(PrefetchStateError::OutsideDomain);
        }
        if let Some(duration) = waited {
            self.report.demand_waits = self.report.demand_waits.saturating_add(1);
            self.report.demand_wait_duration =
                self.report.demand_wait_duration.saturating_add(duration);
        }
        match self.slots.replace(id.clone(), Slot::Empty) {
            Some(Slot::Failed(error)) => Ok(PrefetchDemandResolution::Failed(error)),
            Some(Slot::Completed) => {
                self.report.ready_before_demand = self.report.ready_before_demand.saturating_add(1);
                Ok(PrefetchDemandResolution::Ready)
            }
            _ => Ok(PrefetchDemandResolution::Unscheduled),
        }
    }
    /// Records the first full-queue encounter.
    pub fn begin_backpressure(&mut self) {
        self.report.backpressure_count = self.report.backpressure_count.saturating_add(1);
    }
    /// Records the resolved wait duration.
    pub fn finish_backpressure(&mut self, duration: Duration) {
        self.report.backpressure_duration =
            self.report.backpressure_duration.saturating_add(duration);
    }
    /// Cancels queued work while exact in-flight attempts remain owned.
    pub fn cancel_all(&mut self) -> Result<(), PrefetchStateError<K>> {
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(PrefetchStateError::GenerationExhausted)?;
        self.generation = generation;
        self.report.cancelled = self
            .report
            .cancelled
            .saturating_add(self.queue.len() as u64);
        while let Some(work) = self.queue.pop_front() {
            self.slots.replace(work.id, Slot::Empty);
        }
        Ok(())
    }
    /// Detaches one terminal entry in key order after cancellation is fenced.
    /// Callers can release an external lock before destroying the returned E.
    pub fn take_cancelled_terminal(
        &mut self,
    ) -> Result<Option<(K, PrefetchDemandResolution<E>)>, PrefetchStateError<K>> {
        if !self.is_idle() {
            return Err(PrefetchStateError::CancellationStillInFlight);
        }
        let Some(key) = self.slots.first_terminal().cloned() else {
            return Ok(None);
        };
        let result = match self.slots.replace(key.clone(), Slot::Empty) {
            Some(Slot::Failed(e)) => PrefetchDemandResolution::Failed(e),
            _ => PrefetchDemandResolution::Ready,
        };
        Ok(Some((key, result)))
    }
    /// Abandons completed results, returning the first failure in key order.
    pub fn finish_cancellation(&mut self) -> Result<Option<(K, E)>, PrefetchStateError<K>> {
        let mut first = None;
        while let Some((key, result)) = self.take_cancelled_terminal()? {
            if let PrefetchDemandResolution::Failed(error) = result {
                if first.is_none() {
                    first = Some((key, error));
                }
            }
        }
        Ok(first)
    }
    /// No queued or in-flight operations remain.
    pub fn is_idle(&self) -> bool {
        self.queue.is_empty() && self.active == 0
    }
    /// Current cancellation generation.
    pub const fn generation(&self) -> u64 {
        self.generation
    }
    /// Immutable lifecycle telemetry.
    pub const fn report(&self) -> BackgroundPrefetchReport {
        self.report
    }
}

/// One bounded wake notification predicate, protected by the lifecycle lock.
/// A receiver consumes a token before draining available FIFO work. Cancellation
/// does not reset a pending token: it may still be present in the host channel.
#[derive(Debug, Default)]
pub struct PrefetchWake {
    pending: bool,
}
impl PrefetchWake {
    /// Returns true only for the transition requiring one host notification.
    pub fn request(&mut self) -> bool {
        if self.pending {
            false
        } else {
            self.pending = true;
            true
        }
    }
    /// A receiver has removed the actual notification, or sending it failed.
    pub fn consume(&mut self) {
        self.pending = false;
    }
    /// Whether an actual queued notification remains outstanding.
    pub const fn pending(&self) -> bool {
        self.pending
    }
}

/// Invalid use of the prefetch execution lifecycle.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum PrefetchStateError<K = OffloadUnitId> {
    /// Bounded admission requires at least one queue slot.
    #[error("background prefetch queue capacity must be nonzero")]
    ZeroQueueCapacity,
    /// The selected domain has no such ordinal.
    #[error("prefetch unit is outside the selected domain")]
    OutsideDomain,
    /// A checked attempt sequence cannot be issued again.
    #[error("prefetch attempt sequence exhausted")]
    SequenceExhausted,
    /// A cloned token belongs to an earlier attempt, even in the same generation.
    #[error("prefetch completion attempt does not match active work")]
    CompletionSequenceMismatch,
    /// The monotonic cancellation generation cannot advance safely.
    #[error("background prefetch cancellation generation exhausted")]
    GenerationExhausted,
    /// Backend notification rollback did not refer to admitted queued work.
    #[error("prefetch work {id} generation {generation} is not queued")]
    WorkNotQueued {
        /// Logical unit in the invalid operation.
        id: K,
        /// Exact operation generation.
        generation: u64,
    },
    /// Completion did not refer to backend-submitted work.
    #[error("prefetch work {id} generation {generation} is not in flight")]
    WorkNotInFlight {
        /// Logical unit in the invalid operation.
        id: K,
        /// Exact operation generation.
        generation: u64,
    },
    /// Completion used a different exact generation from the active operation.
    #[error("prefetch completion generation mismatch for {id}: expected {expected}, got {actual}")]
    CompletionGenerationMismatch {
        /// Logical unit in the invalid operation.
        id: K,
        /// Generation owned by the in-flight operation.
        expected: u64,
        /// Generation supplied by the completion.
        actual: u64,
    },
    /// Demand attempted to consume a nonterminal operation.
    #[error("prefetch demand for {id} is still pending")]
    DemandStillPending {
        /// Logical unit whose operation is nonterminal.
        id: K,
    },
    /// Cancellation finalization preceded exact in-flight completion.
    #[error("background prefetch cancellation still owns in-flight work")]
    CancellationStillInFlight,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: &str) -> OffloadUnitId {
        OffloadUnitId::new(value).unwrap()
    }

    #[derive(Debug, Default)]
    struct MockBackend {
        executed: Vec<OffloadUnitId>,
    }

    impl MockBackend {
        fn execute(
            &mut self,
            state: &mut PrefetchExecutionState<&'static str>,
            result: Result<(), &'static str>,
        ) -> PrefetchCompletion {
            let work = state.begin_next().expect("mock backend has admitted work");
            self.executed.push(work.id().clone());
            state.complete(work, result).unwrap()
        }
    }

    #[test]
    fn disjoint_worker_reports_accumulate_actual_completion_and_failure() {
        let mut first = PrefetchExecutionState::new(2).unwrap();
        let a = id("a");
        first.admit(a.clone(), false);
        let work = first.begin_next().unwrap();
        first.complete(work, Ok::<(), &'static str>(())).unwrap();
        first.resolve_demand(&a, Some(Duration::from_millis(7))).unwrap();
        let mut second = PrefetchExecutionState::new(1).unwrap();
        let b = id("b");
        second.admit(b.clone(), false);
        let work = second.begin_next().unwrap();
        second.complete(work, Err("actual failure")).unwrap();
        second.resolve_demand(&b, Some(Duration::from_millis(11))).unwrap();
        let mut report = first.report();
        report.accumulate(second.report());
        assert_eq!((report.submitted(), report.started(), report.completed(), report.failed()), (2, 2, 1, 1));
        assert_eq!((report.queue_capacity(), report.peak_queue_occupancy()), (2, 1));
        assert_eq!(report.demand_waits(), 2);
        assert_eq!(report.demand_wait_duration(), Duration::from_millis(18));
    }

    #[test]
    fn mock_backend_reuses_fifo_admission_coalescing_and_exact_completion() {
        let mut state = PrefetchExecutionState::new(2).unwrap();
        let first = id("layer.0");
        let second = id("layer.1");
        let third = id("layer.2");

        assert!(matches!(
            state.admit(first.clone(), false),
            PrefetchAdmission::Admitted(_)
        ));
        assert_eq!(
            state.admit(first.clone(), false),
            PrefetchAdmission::Coalesced
        );
        assert!(matches!(
            state.admit(second.clone(), false),
            PrefetchAdmission::Admitted(_)
        ));
        assert_eq!(
            state.admit(third.clone(), false),
            PrefetchAdmission::AtCapacity
        );

        let mut backend = MockBackend::default();
        assert_eq!(
            backend.execute(&mut state, Ok(())),
            PrefetchCompletion::Published
        );
        assert!(matches!(
            state.admit(third.clone(), false),
            PrefetchAdmission::Admitted(_)
        ));
        backend.execute(&mut state, Ok(()));
        backend.execute(&mut state, Ok(()));
        assert_eq!(backend.executed, [first, second, third]);
        assert_eq!(state.report().submitted(), 3);
        assert_eq!(state.report().coalesced(), 1);
        assert_eq!(state.report().peak_queue_occupancy(), 2);
    }

    #[test]
    fn cancellation_discards_queue_but_retains_exact_in_flight_ownership() {
        let mut state = PrefetchExecutionState::<()>::new(2).unwrap();
        let active = id("layer.0");
        let queued = id("layer.1");
        state.admit(active.clone(), false);
        state.admit(queued, false);
        let work = state.begin_next().unwrap();

        state.cancel_all().unwrap();
        assert!(!state.is_idle());
        assert!(matches!(
            state.finish_cancellation(),
            Err(PrefetchStateError::CancellationStillInFlight)
        ));
        assert_eq!(
            state.complete(work, Ok(())).unwrap(),
            PrefetchCompletion::Discarded
        );
        assert!(state.is_idle());
        assert_eq!(state.finish_cancellation().unwrap(), None);
        assert_eq!(state.report().cancelled(), 2);
        assert_eq!(state.report().completed(), 0);
        assert_eq!(
            state.observe_demand(&active),
            PrefetchDemandObservation::Unscheduled
        );
    }

    #[test]
    fn failure_is_delivered_once_and_a_new_attempt_supersedes_it() {
        let mut state = PrefetchExecutionState::new(1).unwrap();
        let unit = id("layer.0");
        state.admit(unit.clone(), false);
        let work = state.begin_next().unwrap();
        state.complete(work, Err("disk read failed")).unwrap();
        assert_eq!(
            state.observe_demand(&unit),
            PrefetchDemandObservation::Failed
        );

        assert!(matches!(
            state.admit(unit.clone(), false),
            PrefetchAdmission::Admitted(_)
        ));
        assert_eq!(
            state.observe_demand(&unit),
            PrefetchDemandObservation::Queued
        );
        let work = state.begin_next().unwrap();
        state.complete(work, Ok(())).unwrap();
        assert_eq!(
            state
                .resolve_demand(&unit, Some(Duration::from_millis(3)))
                .unwrap(),
            PrefetchDemandResolution::Ready
        );
        assert_eq!(state.report().failed(), 1);
        assert_eq!(state.report().completed(), 1);
        assert_eq!(state.report().demand_waits(), 1);
    }

    #[test]
    fn rollback_and_residency_observation_preserve_admission_accounting() {
        let mut state = PrefetchExecutionState::<()>::new(1).unwrap();
        let unit = id("layer.0");
        let PrefetchAdmission::Admitted(work) = state.admit(unit.clone(), false) else {
            panic!("missing unit should be admitted");
        };
        state.rollback_admission(&work).unwrap();
        assert_eq!(state.report().submitted(), 0);
        assert!(state.is_idle());

        assert_eq!(
            state.admit(unit.clone(), true),
            PrefetchAdmission::Coalesced
        );
        assert_eq!(
            state.observe_demand(&unit),
            PrefetchDemandObservation::Ready
        );
        assert_eq!(
            state.resolve_demand(&unit, None).unwrap(),
            PrefetchDemandResolution::Ready
        );
        assert_eq!(state.report().ready_before_demand(), 1);
    }

    #[test]
    fn report_serialization_round_trip_preserves_stable_fields() {
        let mut state = PrefetchExecutionState::<()>::new(3).unwrap();
        state.begin_backpressure();
        assert_eq!(state.report().backpressure_count(), 1);
        assert_eq!(state.report().backpressure_duration(), Duration::ZERO);
        state.finish_backpressure(Duration::from_millis(7));
        state.admit(id("layer.0"), false);
        let report = state.report();
        let encoded = serde_json::to_string(&report).unwrap();
        let decoded: BackgroundPrefetchReport = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, report);
        assert_eq!(decoded.queue_capacity(), 3);
        assert_eq!(decoded.backpressure_count(), 1);
        assert_eq!(decoded.backpressure_duration(), Duration::from_millis(7));
    }
}

#[cfg(test)]
mod fixed_tests;

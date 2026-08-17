//! Backend-neutral admission and lifecycle for bounded residency prefetching.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
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

    /// Submissions that waited for admission capacity.
    pub const fn backpressure_count(self) -> u64 {
        self.backpressure_count
    }

    /// Time spent waiting for admission capacity.
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
pub struct PrefetchWork {
    generation: u64,
    id: OffloadUnitId,
}

impl PrefetchWork {
    /// Cancellation generation in which this work was admitted.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Logical residency unit to materialize.
    pub const fn id(&self) -> &OffloadUnitId {
        &self.id
    }
}

/// Result of attempting to admit one prefetch operation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum PrefetchAdmission {
    /// A new operation was admitted and awaits backend execution.
    Admitted(PrefetchWork),
    /// Existing queued, submitted, completed, or resident state satisfies it.
    Coalesced,
    /// The bounded queue cannot admit another operation yet.
    AtCapacity,
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

/// Backend-neutral bounded prefetch admission and execution lifecycle.
///
/// The backend owns workers, buffers, I/O, and completion primitives. This
/// value is the sole owner of FIFO ordering, bounded admission, duplicate
/// coalescing, exact operation generations, cancellation fencing, failure
/// retention and recovery, and lifecycle telemetry.
#[derive(Debug)]
pub struct PrefetchExecutionState<E> {
    generation: u64,
    queue_capacity: usize,
    queue: VecDeque<PrefetchWork>,
    queued: BTreeSet<OffloadUnitId>,
    in_flight: BTreeMap<OffloadUnitId, u64>,
    completed: BTreeSet<OffloadUnitId>,
    failures: BTreeMap<OffloadUnitId, E>,
    report: BackgroundPrefetchReport,
}

impl<E> PrefetchExecutionState<E> {
    /// Creates an empty executor state with a finite, nonzero queue capacity.
    pub fn new(queue_capacity: usize) -> Result<Self, PrefetchStateError> {
        if queue_capacity == 0 {
            return Err(PrefetchStateError::ZeroQueueCapacity);
        }
        Ok(Self {
            generation: 0,
            queue_capacity,
            queue: VecDeque::new(),
            queued: BTreeSet::new(),
            in_flight: BTreeMap::new(),
            completed: BTreeSet::new(),
            failures: BTreeMap::new(),
            report: BackgroundPrefetchReport {
                queue_capacity,
                ..BackgroundPrefetchReport::default()
            },
        })
    }

    /// Admits a missing unit, coalesces existing work, or reports backpressure.
    ///
    /// `resident` is supplied by the backend after checking its concrete
    /// storage. A completed logical result that is no longer resident is
    /// counted as evicted-before-use and is eligible for a new attempt.
    pub fn admit(&mut self, id: OffloadUnitId, resident: bool) -> PrefetchAdmission {
        if self.queued.contains(&id) || self.in_flight.contains_key(&id) {
            self.report.coalesced = self.report.coalesced.saturating_add(1);
            return PrefetchAdmission::Coalesced;
        }

        if self.completed.contains(&id) && !resident {
            self.completed.remove(&id);
            self.report.evicted_before_use = self.report.evicted_before_use.saturating_add(1);
        }
        if resident {
            self.failures.remove(&id);
            self.completed.insert(id);
            self.report.coalesced = self.report.coalesced.saturating_add(1);
            return PrefetchAdmission::Coalesced;
        }
        if self.queue.len() == self.queue_capacity {
            return PrefetchAdmission::AtCapacity;
        }

        // A new explicit attempt supersedes a previously observed backend
        // failure. Demand will see the result of this exact generation.
        self.failures.remove(&id);
        let work = PrefetchWork {
            generation: self.generation,
            id,
        };
        self.queued.insert(work.id.clone());
        self.queue.push_back(work.clone());
        self.report.submitted = self.report.submitted.saturating_add(1);
        self.report.peak_queue_occupancy = self.report.peak_queue_occupancy.max(self.queue.len());
        PrefetchAdmission::Admitted(work)
    }

    /// Rolls back one admitted operation whose backend notification failed.
    pub fn rollback_admission(&mut self, work: &PrefetchWork) -> Result<(), PrefetchStateError> {
        let Some(position) = self.queue.iter().position(|queued| queued == work) else {
            return Err(PrefetchStateError::WorkNotQueued {
                id: work.id.clone(),
                generation: work.generation,
            });
        };
        self.queue.remove(position);
        self.queued.remove(&work.id);
        self.report.submitted = self.report.submitted.saturating_sub(1);
        Ok(())
    }

    /// Selects the oldest admitted operation for backend submission.
    pub fn begin_next(&mut self) -> Option<PrefetchWork> {
        let work = self.queue.pop_front()?;
        self.queued.remove(&work.id);
        self.in_flight.insert(work.id.clone(), work.generation);
        self.report.started = self.report.started.saturating_add(1);
        Some(work)
    }

    /// Applies one exact backend completion.
    pub fn complete(
        &mut self,
        work: PrefetchWork,
        result: Result<(), E>,
    ) -> Result<PrefetchCompletion, PrefetchStateError> {
        let Some(active_generation) = self.in_flight.get(&work.id).copied() else {
            return Err(PrefetchStateError::WorkNotInFlight {
                id: work.id,
                generation: work.generation,
            });
        };
        if active_generation != work.generation {
            return Err(PrefetchStateError::CompletionGenerationMismatch {
                id: work.id,
                expected: active_generation,
                actual: work.generation,
            });
        }
        self.in_flight.remove(&work.id);
        if work.generation != self.generation {
            self.report.cancelled = self.report.cancelled.saturating_add(1);
            return Ok(PrefetchCompletion::Discarded);
        }
        match result {
            Ok(()) => {
                self.completed.insert(work.id);
                self.report.completed = self.report.completed.saturating_add(1);
                Ok(PrefetchCompletion::Published)
            }
            Err(error) => {
                self.failures.insert(work.id, error);
                self.report.failed = self.report.failed.saturating_add(1);
                Ok(PrefetchCompletion::Failed)
            }
        }
    }

    /// Observes current background ownership when demand first arrives.
    pub fn observe_demand(&mut self, id: &OffloadUnitId) -> PrefetchDemandObservation {
        if self.queued.contains(id) {
            PrefetchDemandObservation::Queued
        } else if self.in_flight.contains_key(id) {
            self.report.in_flight_at_demand = self.report.in_flight_at_demand.saturating_add(1);
            PrefetchDemandObservation::InFlight
        } else if self.failures.contains_key(id) {
            PrefetchDemandObservation::Failed
        } else if self.completed.contains(id) {
            PrefetchDemandObservation::Ready
        } else {
            PrefetchDemandObservation::Unscheduled
        }
    }

    /// Whether an admitted or submitted operation still owns this unit.
    pub fn is_pending(&self, id: &OffloadUnitId) -> bool {
        self.queued.contains(id) || self.in_flight.contains_key(id)
    }

    /// Consumes the terminal background result for one demand acquisition.
    pub fn resolve_demand(
        &mut self,
        id: &OffloadUnitId,
        waited: Option<Duration>,
    ) -> Result<PrefetchDemandResolution<E>, PrefetchStateError> {
        if self.is_pending(id) {
            return Err(PrefetchStateError::DemandStillPending { id: id.clone() });
        }
        if let Some(duration) = waited {
            self.report.demand_waits = self.report.demand_waits.saturating_add(1);
            self.report.demand_wait_duration =
                self.report.demand_wait_duration.saturating_add(duration);
        }
        if let Some(error) = self.failures.remove(id) {
            return Ok(PrefetchDemandResolution::Failed(error));
        }
        if self.completed.remove(id) {
            self.report.ready_before_demand = self.report.ready_before_demand.saturating_add(1);
            return Ok(PrefetchDemandResolution::Ready);
        }
        Ok(PrefetchDemandResolution::Unscheduled)
    }

    /// Records one submission that waited for bounded admission.
    pub fn record_backpressure(&mut self, duration: Duration) {
        self.report.backpressure_count = self.report.backpressure_count.saturating_add(1);
        self.report.backpressure_duration =
            self.report.backpressure_duration.saturating_add(duration);
    }

    /// Cancels all queued work immediately and fences exact in-flight work.
    ///
    /// In-flight backend resources remain owned until [`Self::complete`] sees
    /// their exact operation. Queued work needs no backend cancellation and is
    /// discarded synchronously.
    pub fn cancel_all(&mut self) -> Result<(), PrefetchStateError> {
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or(PrefetchStateError::GenerationExhausted)?;
        self.report.cancelled = self
            .report
            .cancelled
            .saturating_add(self.queue.len() as u64);
        self.queue.clear();
        self.queued.clear();
        Ok(())
    }

    /// Finishes cancellation after every exact in-flight operation resolved.
    ///
    /// Completed prefetches are abandoned. The first retained backend failure
    /// is returned in deterministic logical-unit order.
    pub fn finish_cancellation(
        &mut self,
    ) -> Result<Option<(OffloadUnitId, E)>, PrefetchStateError> {
        if !self.is_idle() {
            return Err(PrefetchStateError::CancellationStillInFlight);
        }
        self.completed.clear();
        let failure = self.failures.pop_first();
        self.failures.clear();
        Ok(failure)
    }

    /// Whether no admitted or backend-submitted operation remains.
    pub fn is_idle(&self) -> bool {
        self.queue.is_empty() && self.in_flight.is_empty()
    }

    /// Current cancellation generation.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Immutable telemetry snapshot.
    pub const fn report(&self) -> BackgroundPrefetchReport {
        self.report
    }
}

/// Invalid use of the prefetch execution lifecycle.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum PrefetchStateError {
    /// Bounded admission requires at least one queue slot.
    #[error("background prefetch queue capacity must be nonzero")]
    ZeroQueueCapacity,
    /// The monotonic cancellation generation cannot advance safely.
    #[error("background prefetch cancellation generation exhausted")]
    GenerationExhausted,
    /// Backend notification rollback did not refer to admitted queued work.
    #[error("prefetch work {id} generation {generation} is not queued")]
    WorkNotQueued {
        /// Logical unit in the invalid operation.
        id: OffloadUnitId,
        /// Exact operation generation.
        generation: u64,
    },
    /// Completion did not refer to backend-submitted work.
    #[error("prefetch work {id} generation {generation} is not in flight")]
    WorkNotInFlight {
        /// Logical unit in the invalid operation.
        id: OffloadUnitId,
        /// Exact operation generation.
        generation: u64,
    },
    /// Completion used a different exact generation from the active operation.
    #[error("prefetch completion generation mismatch for {id}: expected {expected}, got {actual}")]
    CompletionGenerationMismatch {
        /// Logical unit in the invalid operation.
        id: OffloadUnitId,
        /// Generation owned by the in-flight operation.
        expected: u64,
        /// Generation supplied by the completion.
        actual: u64,
    },
    /// Demand attempted to consume a nonterminal operation.
    #[error("prefetch demand for {id} is still pending")]
    DemandStillPending {
        /// Logical unit whose operation is nonterminal.
        id: OffloadUnitId,
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
        state.record_backpressure(Duration::from_millis(7));
        state.admit(id("layer.0"), false);
        let report = state.report();
        let encoded = serde_json::to_string(&report).unwrap();
        let decoded: BackgroundPrefetchReport = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, report);
        assert_eq!(decoded.queue_capacity(), 3);
        assert_eq!(decoded.backpressure_duration(), Duration::from_millis(7));
    }
}

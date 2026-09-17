use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
#[allow(dead_code)]
mod legacy;
fn id(n: usize) -> OffloadUnitId {
    OffloadUnitId::new(format!("unit.{n}")).unwrap()
}

#[test]
fn fixed_and_ordinary_transitions_match_independent_old_worker() {
    let mut fixed = PrefetchExecutionState::<&str, usize>::with_unit_count(7, 2).unwrap();
    let mut ordinary = PrefetchExecutionState::<&str>::new(2).unwrap();
    let mut old = legacy::PrefetchExecutionState::<&str>::new(2).unwrap();
    let capacities = fixed.fixed_capacities();
    // N terminal owners are intentionally larger than Q. Completion, failure,
    // resident coalescing, eviction, pending demand and cancellation all use
    // the independent pre-refactor transition source, not another new worker.
    for cycle in 0..20 {
        for n in 0..7 {
            assert!(matches!(
                fixed.admit(n, false),
                PrefetchAdmission::Admitted(_)
            ));
            assert!(matches!(
                ordinary.admit(id(n), false),
                PrefetchAdmission::Admitted(_)
            ));
            assert!(matches!(
                old.admit(id(n), false),
                legacy::PrefetchAdmission::Admitted(_)
            ));
            assert_eq!(
                format!("{:?}", fixed.observe_demand(&n)),
                format!("{:?}", old.observe_demand(&id(n)))
            );
            ordinary.observe_demand(&id(n));
            assert!(matches!(
                fixed.admit(n, false),
                PrefetchAdmission::Coalesced
            ));
            ordinary.admit(id(n), false);
            old.admit(id(n), false);
            let f = fixed.begin_next().unwrap();
            let o = ordinary.begin_next().unwrap();
            let l = old.begin_next().unwrap();
            let result = if (n + cycle) % 3 == 0 {
                Err("read failed")
            } else {
                Ok(())
            };
            assert_eq!(
                format!("{:?}", fixed.complete(f, result).unwrap()),
                format!("{:?}", old.complete(l, result).unwrap())
            );
            ordinary.complete(o, result).unwrap();
        }
        for n in [6, 2, 4] {
            assert_eq!(
                format!(
                    "{:?}",
                    fixed
                        .resolve_demand(&n, Some(Duration::from_millis(3)))
                        .unwrap()
                ),
                format!(
                    "{:?}",
                    old.resolve_demand(&id(n), Some(Duration::from_millis(3)))
                        .unwrap()
                )
            );
            ordinary
                .resolve_demand(&id(n), Some(Duration::from_millis(3)))
                .unwrap();
        }
        for n in [0, 1, 2] {
            let resident = n == 0;
            let f = fixed.admit(n, resident);
            let o = ordinary.admit(id(n), resident);
            let l = old.admit(id(n), resident);
            assert_eq!(
                matches!(f, PrefetchAdmission::Coalesced),
                matches!(l, legacy::PrefetchAdmission::Coalesced)
            );
            assert_eq!(
                matches!(o, PrefetchAdmission::AtCapacity),
                matches!(l, legacy::PrefetchAdmission::AtCapacity)
            );
        }
        fixed.cancel_all().unwrap();
        ordinary.cancel_all().unwrap();
        old.cancel_all().unwrap();
        let f = fixed
            .finish_cancellation()
            .unwrap()
            .map(|(n, e)| (id(n), e));
        assert_eq!(f, old.finish_cancellation().unwrap());
        assert_eq!(f, ordinary.finish_cancellation().unwrap());
        assert_eq!(
            serde_json::to_value(fixed.report()).unwrap(),
            serde_json::to_value(old.report()).unwrap()
        );
        assert_eq!(fixed.report(), ordinary.report());
        assert_eq!(fixed.fixed_capacities(), capacities);
    }
    // The ordinary public state still permits multiple concurrent operations.
    // The actual selected single-worker representation deliberately does not.
    for n in 0..2 {
        ordinary.admit(id(n), false);
        old.admit(id(n), false);
    }
    let a = ordinary.begin_next().unwrap();
    let b = ordinary.begin_next().unwrap();
    let old_a = old.begin_next().unwrap();
    let old_b = old.begin_next().unwrap();
    ordinary.complete(b, Err("second")).unwrap();
    old.complete(old_b, Err("second")).unwrap();
    ordinary.complete(a, Ok(())).unwrap();
    old.complete(old_a, Ok(())).unwrap();
    assert_eq!(
        ordinary.finish_cancellation().unwrap(),
        old.finish_cancellation().unwrap()
    );
    assert_eq!(
        serde_json::to_value(ordinary.report()).unwrap(),
        serde_json::to_value(old.report()).unwrap()
    );
}

#[test]
fn fixed_domain_fifo_and_single_flight_never_grow() {
    let mut state = PrefetchExecutionState::<(), usize>::with_unit_count(8, 2).unwrap();
    let caps = state.fixed_capacities();
    for n in 0..100 {
        let a = n % 8;
        let b = (n + 1) % 8;
        let c = (n + 2) % 8;
        state.admit(a, false);
        state.admit(b, false);
        assert_eq!(state.admit(c, false), PrefetchAdmission::AtCapacity);
        let first = state.begin_next().unwrap();
        assert_eq!(*first.id(), a);
        assert!(state.begin_next().is_none());
        state.complete(first, Ok(())).unwrap();
        let second = state.begin_next().unwrap();
        assert_eq!(*second.id(), b);
        state.complete(second, Ok(())).unwrap();
        state.finish_cancellation().unwrap();
        assert_eq!(state.fixed_capacities(), caps);
    }
    let before = state.report();
    assert_eq!(
        state.admit(8, false),
        PrefetchAdmission::Rejected(PrefetchStateError::OutsideDomain)
    );
    assert_eq!(state.report(), before);
}

#[test]
fn each_actual_fixed_reserve_failure_retains_prior_capacity() {
    for cause in [PrefetchStorageCause::Slots, PrefetchStorageCause::Queue] {
        let error =
            PrefetchExecutionState::<(), usize>::prepare_fixed(8, 3, Some(cause)).unwrap_err();
        assert_eq!(error.cause(), cause);
        assert!(std::error::Error::source(&error).is_some());
        let (slots, queue) = error.retained_capacities();
        assert_eq!(queue, 0);
        if cause == PrefetchStorageCause::Slots {
            assert_eq!(slots, 0);
        } else {
            assert!(slots >= 8);
        }
    }
    assert_eq!(
        PrefetchExecutionState::<(), usize>::with_unit_count(1, 0)
            .unwrap_err()
            .cause(),
        PrefetchStorageCause::ZeroQueueCapacity
    );
    let empty = PrefetchExecutionState::<(), usize>::with_unit_count(0, 1).unwrap();
    assert_eq!(empty.fixed_capacities().0, 0);
}

#[derive(Debug)]
struct ErrorOwner(Arc<AtomicUsize>);
impl Drop for ErrorOwner {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
#[test]
fn same_generation_stale_attempt_cannot_complete_a_retry_or_destroy_its_result() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut state = PrefetchExecutionState::<ErrorOwner, usize>::with_unit_count(1, 1).unwrap();
    state.admit(0, false);
    let first = state.begin_next().unwrap();
    let stale = first.clone();
    state.complete(first, Ok(())).unwrap();
    state.resolve_demand(&0, None).unwrap();
    state.admit(0, false);
    let second = state.begin_next().unwrap();
    assert_eq!(stale.generation(), second.generation());
    assert_ne!(stale.sequence(), second.sequence());
    let (error, result) = state
        .complete_retaining(stale, Err(ErrorOwner(drops.clone())))
        .unwrap_err();
    assert_eq!(error, PrefetchStateError::CompletionSequenceMismatch);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(state.is_pending(&0));
    drop(result);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    state.complete(second, Ok(())).unwrap();
    let mut queued = PrefetchExecutionState::<(), usize>::with_unit_count(1, 1).unwrap();
    let PrefetchAdmission::Admitted(first) = queued.admit(0, false) else {
        unreachable!()
    };
    queued.rollback_admission(&first).unwrap();
    let PrefetchAdmission::Admitted(second) = queued.admit(0, false) else {
        unreachable!()
    };
    assert_ne!(first.sequence(), second.sequence());
    let before = queued.report();
    assert!(matches!(
        queued.rollback_admission(&first),
        Err(PrefetchStateError::WorkNotQueued { .. })
    ));
    assert_eq!(queued.report(), before);
    queued.rollback_admission(&second).unwrap();
}

#[test]
fn last_issued_attempt_finishes_and_future_exhaustion_is_atomic() {
    let mut state = PrefetchExecutionState::<&str, usize>::with_unit_count(2, 2).unwrap();
    state.next_sequence = Some(u64::MAX);
    state.admit(0, false);
    let work = state.begin_next().unwrap();
    state.complete(work, Err("retained")).unwrap();
    let before = state.report();
    assert_eq!(
        state.admit(0, false),
        PrefetchAdmission::Rejected(PrefetchStateError::SequenceExhausted)
    );
    assert_eq!(state.report(), before);
    assert_eq!(
        state.resolve_demand(&0, None).unwrap(),
        PrefetchDemandResolution::Failed("retained")
    );
    assert_eq!(state.admit(1, true), PrefetchAdmission::Coalesced);
    let before = state.report();
    assert_eq!(
        state.admit(1, false),
        PrefetchAdmission::Rejected(PrefetchStateError::SequenceExhausted)
    );
    assert_eq!(state.report(), before);
    assert_eq!(state.observe_demand(&1), PrefetchDemandObservation::Ready);
    state.generation = u64::MAX;
    assert_eq!(
        state.cancel_all(),
        Err(PrefetchStateError::GenerationExhausted)
    );
    assert_eq!(state.observe_demand(&1), PrefetchDemandObservation::Ready);
}

#[test]
fn fixed_terminal_and_stale_failure_owners_retire_only_when_transferred_or_dropped() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut state = PrefetchExecutionState::<ErrorOwner, usize>::with_unit_count(3, 1).unwrap();
    for n in 0..3 {
        state.admit(n, false);
        let w = state.begin_next().unwrap();
        state.complete(w, Err(ErrorOwner(drops.clone()))).unwrap();
    }
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let (a, retired) = state.admit_retaining(1, false);
    assert!(matches!(a, PrefetchAdmission::Admitted(_)));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(retired);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    let work = state.begin_next().unwrap();
    state.cancel_all().unwrap();
    let (outcome, retired) = state
        .complete_retaining(work, Err(ErrorOwner(drops.clone())))
        .unwrap();
    assert_eq!(outcome, PrefetchCompletion::Discarded);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    let (key, first) = state.take_cancelled_terminal().unwrap().unwrap();
    assert_eq!(key, 0);
    drop(state);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    drop(first);
    assert_eq!(drops.load(Ordering::SeqCst), 3);
    drop(retired);
    assert_eq!(drops.load(Ordering::SeqCst), 4);
}

#[test]
fn wake_cancellation_does_not_forge_consumption_of_an_outstanding_token() {
    let mut wake = PrefetchWake::default();
    let mut sent = 0;
    let mut state = PrefetchExecutionState::<(), usize>::with_unit_count(1, 1).unwrap();
    for _ in 0..1000 {
        state.admit(0, false);
        sent += usize::from(wake.request());
        state.cancel_all().unwrap();
    }
    assert_eq!(sent, 1);
    assert!(wake.pending());
    wake.consume();
    assert!(!wake.pending());
    assert!(wake.request());
    assert!(!wake.request());
}

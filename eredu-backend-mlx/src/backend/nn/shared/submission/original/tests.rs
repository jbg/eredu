use super::*;
use crate::backend::{ordinary_retirement, submission_recovery};
use std::time::{Duration, Instant};

thread_local! { pub(super) static FAIL_RESERVE: Cell<Option<usize>> = const { Cell::new(None) }; }

#[derive(Clone)]
struct Custody(Rc<Cell<usize>>);
impl Drop for Custody {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}
struct State {
    settled: Cell<bool>,
    outcome: Cell<ScopedSubmissionProgress>,
    payload_dropped: Rc<Cell<bool>>,
    late_drains: Cell<usize>,
}
#[derive(Clone)]
struct Fake(Rc<State>);
impl Observer for Fake {
    type Error = u8;
    fn observe(&self) -> Result<Observation, u8> {
        Ok(Observation {
            outcome: self.0.outcome.get(),
            status: Status {
                settled: self.0.settled.get(),
                failed: false,
                blocked: false,
            },
        })
    }
    fn retire_terminal(&self) -> Result<safemlx::SubmissionRetirement, u8> {
        if self.0.payload_dropped.get() {
            self.0.late_drains.set(self.0.late_drains.get() + 1);
        }
        Ok(safemlx::SubmissionRetirement::CompleteSnapshot)
    }
}
pub(super) struct PayloadWitness(pub(super) Rc<Cell<bool>>);
impl Drop for PayloadWitness {
    fn drop(&mut self) {
        assert!(safemlx::can_reclaim_submission_resources());
        self.0.set(true);
    }
}
type Prepared = PreparedNeuralSubmission<Custody, Fake>;

fn fixture(consumers: usize, settled: bool) -> (Prepared, Fake, Rc<Cell<usize>>) {
    let dropped = Rc::new(Cell::new(false));
    let guard_drops = Rc::new(Cell::new(0));
    let prepared = Prepared::try_new(
        NeuralSubmissionShape::new(2, consumers).unwrap(),
        Custody(Rc::clone(&guard_drops)),
    )
    .unwrap();
    prepared
        .shared
        .payload
        .borrow_mut()
        .as_mut()
        .unwrap()
        .witness = Some(PayloadWitness(Rc::clone(&dropped)));
    (
        prepared,
        Fake(Rc::new(State {
            settled: Cell::new(settled),
            outcome: Cell::new(ScopedSubmissionProgress::Observed),
            payload_dropped: dropped,
            late_drains: Cell::new(0),
        })),
        guard_drops,
    )
}
fn reap_until(mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !done() {
        submission_recovery::reap();
        assert!(
            Instant::now() < deadline,
            "source fixture retirement timed out"
        );
        std::thread::yield_now();
    }
}
fn ordinary_until(mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !done() {
        submission_recovery::reap();
        ordinary_retirement::reclaim();
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
}

#[test]
fn neural_prepared_zero_and_exact_shapes_preserve_requested_storage_and_checked_layout() {
    let zero = NeuralSubmissionShape::new(0, 0).unwrap();
    assert_eq!(zero.host_resources(), 0);
    assert!(Prepared::control_bytes(zero).unwrap() > 0);
    assert!(NeuralSubmissionShape::new(0, usize::MAX).is_none());
    assert!(NeuralSubmissionShape::new(usize::MAX, 0).is_none());
    let (prepared, observer, _) = fixture(3, true);
    {
        let payload = prepared.shared.payload.borrow();
        let payload = payload.as_ref().unwrap();
        assert_eq!(payload.arrays.len(), 0);
        assert!(payload.arrays.capacity() >= 2);
        assert_eq!(payload.clone_slots.len(), 2);
        assert_eq!(payload.cleanups.len(), 4);
        assert!(payload.cleanups.iter().all(Option::is_none));
        assert_eq!(prepared.consumers.len(), 3);
        assert!(prepared
            .consumers
            .iter()
            .all(|slot| matches!(slot, ConsumerSlot::Ready(_))));
    }
    drop(prepared);
    ordinary_until(|| observer.0.payload_dropped.get());
}

#[test]
fn neural_prepared_actual_reserve_failures_retain_each_constructed_prefix_and_custody() {
    for (site, expected) in ["arrays", "cleanup slots", "consumer slots", "clone slots"]
        .into_iter()
        .enumerate()
    {
        let drops = Rc::new(Cell::new(0));
        FAIL_RESERVE.with(|fail| fail.set(Some(site)));
        let error = Prepared::try_new(
            NeuralSubmissionShape::new(2, 2).unwrap(),
            Custody(Rc::clone(&drops)),
        )
        .unwrap_err();
        let SubmissionPreparationError {
            cause,
            pending,
            controls,
        } = error;
        match cause {
            storage::SubmissionPreparationCause::Reserve { site, cause } => {
                assert_eq!(site, expected);
                // Real Vec::try_reserve_exact(usize::MAX) CapacityOverflow.
                assert!(!cause.to_string().is_empty());
            }
            other => panic!("unexpected {other:?}"),
        }
        let prepared = pending.unwrap();
        {
            let payload = prepared.shared.payload.borrow();
            let payload = payload.as_ref().unwrap();
            if site > 0 {
                assert!(payload.arrays.capacity() >= 2);
            }
            if site > 1 {
                assert_eq!(payload.cleanups.len(), 3);
            }
        }
        assert!(prepared.consumers.is_empty());
        let before = drops.get();
        assert!(Rc::strong_count(&controls.0) >= 5);
        drop(prepared);
        drop(controls);
        assert!(drops.get() > before);
        ordinary_until(|| Rc::strong_count(&drops) == 1);
    }
}

#[test]
fn neural_consumers_consume_fixed_slots_once_without_changing_rc_or_buffer_identity() {
    let (prepared, observer, _) = fixture(2, true);
    let shared_identity = Rc::as_ptr(&prepared.shared);
    let owner = Rc::downgrade(&prepared.shared);
    let consumer_buffer = prepared.consumers.as_ptr();
    let capacity = prepared.consumers.capacity();
    let completion = prepared.activate(observer.clone());
    assert_eq!(
        Rc::as_ptr(&completion.retained.retention().shared),
        shared_identity
    );
    for index in 0..2 {
        let (actual, child) = completion.checkout_consumer(observer.clone()).unwrap();
        assert_eq!(actual, index);
        assert_eq!(child.retention().cleanup_index, index + 1);
        assert_eq!(Rc::as_ptr(&child.retention().shared), shared_identity);
        completion.restore_consumer(index, child);
    }
    assert!(completion.checkout_consumer(observer.clone()).is_none());
    let slots = completion.consumers.borrow();
    assert_eq!(slots.as_ptr(), consumer_buffer);
    assert_eq!(slots.capacity(), capacity);
    assert_eq!(completion.retained.retention().shared.children.get(), 2);
    drop(slots);
    drop(completion);
    reap_until(|| owner.upgrade().is_none());
    ordinary_until(|| observer.0.payload_dropped.get());
    reap_until(|| observer.0.late_drains.get() == 3);
}

#[test]
fn neural_last_consumer_alias_defers_payload_then_same_node_cleanup_after_primary_drop() {
    let (prepared, observer, _) = fixture(1, false);
    let owner = Rc::downgrade(&prepared.shared);
    let completion = prepared.activate(observer.clone());
    let (index, child) = completion.checkout_consumer(observer.clone()).unwrap();
    completion.restore_consumer(index, child);
    drop(completion);
    submission_recovery::reap();
    assert!(owner.upgrade().is_some());
    assert!(!observer.0.payload_dropped.get());
    observer.0.settled.set(true);
    reap_until(|| owner.upgrade().is_none());
    assert!(!observer.0.payload_dropped.get());
    assert_eq!(observer.0.late_drains.get(), 0);
    ordinary_until(|| observer.0.payload_dropped.get());
    reap_until(|| observer.0.late_drains.get() == 2);
}

#[test]
fn neural_targeted_release_destroys_exact_shared_payload_before_final_drain() {
    let (prepared, observer, _) = fixture(0, true);
    let completion = prepared.activate(observer.clone());
    let OriginalNeuralSubmissionCompletion {
        retained,
        consumers,
        next_consumer: _,
        _controls,
    } = completion;
    drop(consumers);
    // Enter the successful handoff only when this test owns the runtime loan;
    // real bounded release itself must occur after that loan ends.
    let mut retained = Some(retained);
    let completed = loop {
        if let Some(completed) = safemlx::try_with_submission_retirement(|| {
            retained.take().unwrap().finish_retaining().unwrap()
        }) {
            break completed;
        }
        std::thread::yield_now();
    };
    assert!(!observer.0.payload_dropped.get());
    assert_eq!(Rc::strong_count(&completed.retention().shared), 1);
    let result =
        completed.release_with(release_payload::<Custody> as fn(&mut SubmissionRetention<Custody>));
    // A foreign runtime owner can race the final exact drain. Either success
    // or retained Busy still destroys this payload only at the unlocked call.
    assert!(observer.0.payload_dropped.get());
    drop(result);
    reap_until(|| observer.0.late_drains.get() == 1);
}

#[test]
fn neural_fixed_consumer_refusals_keep_active_slots_without_poisoning_native_status() {
    let (prepared, observer, _) = fixture(3, true);
    let completion = prepared.activate(observer.clone());
    for (code, outcome) in [
        (8, ScopedSubmissionProgress::NeedsFundedProgress),
        (9, ScopedSubmissionProgress::Unobservable),
        (10, ScopedSubmissionProgress::Busy),
    ] {
        let (index, child) = completion.checkout_consumer(observer.clone()).unwrap();
        assert_eq!(
            completion.publish_consumer_result(index, child, Err(code)),
            Err(code)
        );
        observer.0.outcome.set(outcome);
        // Hold the actual runtime observation loan so foreign test activity
        // cannot replace this deliberately selected fixed outcome with Busy.
        let deadline = Instant::now() + Duration::from_secs(10);
        let observed = loop {
            if let Some(observed) =
                safemlx::try_with_submission_retirement(|| completion.retained.progress().unwrap())
            {
                break observed;
            }
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        };
        assert_eq!(observed.outcome, outcome);
        assert!(!observed.can_retire());
        assert!(!completion.retained.retention().shared.failed.get());
        assert!(matches!(
            completion.consumers.borrow()[index],
            ConsumerSlot::Active(_)
        ));
    }
    observer.0.outcome.set(ScopedSubmissionProgress::Observed);
    let owner = Rc::downgrade(&completion.retained.retention().shared);
    drop(completion);
    reap_until(|| owner.upgrade().is_none());
    ordinary_until(|| observer.0.payload_dropped.get());
    reap_until(|| observer.0.late_drains.get() == 4);
}

use super::*;
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    time::{Duration, Instant},
};

type Log = Rc<RefCell<Vec<&'static str>>>;
struct Custody(Log);
impl Drop for Custody {
    fn drop(&mut self) {
        self.0.borrow_mut().push("custody");
    }
}
struct Payload {
    log: Log,
    observations: Rc<Cell<usize>>,
    settled: Rc<Cell<bool>>,
    token: u32,
}
impl Retention for Payload {
    fn observe(&self, status: Status) {
        self.observations.set(self.observations.get() + 1);
        self.settled.set(status.settled);
        assert!(!status.failed && !status.blocked);
    }
}
impl Drop for Payload {
    fn drop(&mut self) {
        self.log.borrow_mut().push("payload");
    }
}
#[derive(Debug)]
struct NativeCause(Rc<()>);
struct Fake {
    mode: Rc<Cell<u8>>,
    calls: Rc<Cell<usize>>,
    cause: Rc<()>,
    log: Log,
    retirement: Rc<Cell<u8>>,
    retire_calls: Rc<Cell<usize>>,
    map_error: Rc<Cell<bool>>,
}
impl Observer for Fake {
    type Error = NativeCause;
    fn observation_error(&self, observed: Observation) -> Option<NativeCause> {
        if self.map_error.get()
            && (observed.outcome != ScopedSubmissionProgress::Observed
                || observed.status.failed
                || observed.status.blocked)
        {
            assert!(
                !self.log.borrow().contains(&"observer"),
                "map actual cause before observer destruction"
            );
            Some(NativeCause(self.cause.clone()))
        } else {
            None
        }
    }
    fn observe(&self) -> Result<Observation, NativeCause> {
        self.calls.set(self.calls.get() + 1);
        let outcome = match self.mode.get() {
            0 | 1 | 6 | 7 => ScopedSubmissionProgress::Observed,
            2 => ScopedSubmissionProgress::Busy,
            3 => ScopedSubmissionProgress::NeedsFundedProgress,
            4 => ScopedSubmissionProgress::Unobservable,
            5 => return Err(NativeCause(self.cause.clone())),
            _ => unreachable!(),
        };
        // Mode 6 supplies one genuine pending report, followed by terminal.
        let settled = self.mode.get() != 0 && (self.mode.get() != 6 || self.calls.get() > 1);
        Ok(Observation {
            outcome,
            status: Status {
                settled,
                failed: self.mode.get() == 7,
                blocked: false,
            },
        })
    }
    fn retire_terminal(&self) -> Result<safemlx::SubmissionRetirement, NativeCause> {
        self.retire_calls.set(self.retire_calls.get() + 1);
        assert!(
            !safemlx::can_reclaim_submission_resources(),
            "native retirement stays under no-hooks guard"
        );
        match self.retirement.get() {
            0 => Ok(safemlx::SubmissionRetirement::CompleteSnapshot),
            1 => Ok(safemlx::SubmissionRetirement::Busy),
            2 => Err(NativeCause(self.cause.clone())),
            3 => panic!("terminal retirement callback fixture"),
            4 => {
                self.log.borrow_mut().push("retire");
                Ok(safemlx::SubmissionRetirement::CompleteSnapshot)
            }
            _ => unreachable!(),
        }
    }
}
impl Drop for Fake {
    fn drop(&mut self) {
        self.log.borrow_mut().push("observer");
    }
}
struct Fixture {
    mode: Rc<Cell<u8>>,
    calls: Rc<Cell<usize>>,
    observations: Rc<Cell<usize>>,
    settled: Rc<Cell<bool>>,
    cause: Rc<()>,
    log: Log,
    retirement: Rc<Cell<u8>>,
    retire_calls: Rc<Cell<usize>>,
    map_error: Rc<Cell<bool>>,
}
impl Fixture {
    fn new(mode: u8) -> Self {
        Self {
            mode: Rc::new(Cell::new(mode)),
            calls: Rc::new(Cell::new(0)),
            observations: Rc::new(Cell::new(0)),
            settled: Rc::new(Cell::new(false)),
            cause: Rc::new(()),
            log: Rc::new(RefCell::new(Vec::with_capacity(8))),
            retirement: Rc::new(Cell::new(0)),
            retire_calls: Rc::new(Cell::new(0)),
            map_error: Rc::new(Cell::new(false)),
        }
    }
    fn payload(&self) -> Payload {
        Payload {
            log: self.log.clone(),
            observations: self.observations.clone(),
            settled: self.settled.clone(),
            token: 19,
        }
    }
    fn observer(&self) -> Fake {
        Fake {
            mode: self.mode.clone(),
            calls: self.calls.clone(),
            cause: self.cause.clone(),
            log: self.log.clone(),
            retirement: self.retirement.clone(),
            retire_calls: self.retire_calls.clone(),
            map_error: self.map_error.clone(),
        }
    }
    fn prepared(&self) -> PreparedObservedRecovery<Payload, Custody, Fake> {
        PreparedObservedRecovery::new(Custody(self.log.clone()))
    }
    fn retire(&self) {
        let until = Instant::now() + Duration::from_secs(5);
        while !self.log.borrow().contains(&"custody") {
            reap();
            assert!(Instant::now() < until, "observed fixture did not retire");
            std::thread::yield_now();
        }
    }
}
fn guarded<R>(action: impl FnOnce() -> R) -> R {
    let until = Instant::now() + Duration::from_secs(5);
    let mut action = Some(action);
    loop {
        if let Some(result) = safemlx::try_with_submission_retirement(|| action.take().unwrap()()) {
            return result;
        }
        assert!(Instant::now() < until, "fixture runtime loan stayed busy");
        std::thread::yield_now();
    }
}

#[test]
fn prepared_observer_failure_keeps_same_node_payload_and_custody_for_retry() {
    let f = Fixture::new(1);
    let ready = f.prepared();
    let identity = ready.allocation_identity();
    let error =
        match ready.try_activate(f.payload(), || Err::<Fake, _>(NativeCause(f.cause.clone()))) {
            Err(error) => error,
            Ok(_) => panic!("injected acquisition must fail"),
        };
    assert!(Rc::ptr_eq(&error.cause.0, &f.cause));
    assert_eq!(error.pending.allocation_identity(), identity);
    assert_eq!(error.retention.token, 19);
    assert_eq!(f.calls.get(), 0);
    assert_eq!(f.observations.get(), 0);
    assert!(f.log.borrow().is_empty());
    let ObservedActivationError {
        cause,
        retention,
        pending,
    } = error;
    drop(cause);
    let mut active = pending.activate(retention, f.observer());
    assert_eq!(active.allocation_identity(), identity);
    active.retention_mut().token = 23;
    assert_eq!(active.retention().token, 23);
    let unboxed = UNBOXED.with(Cell::get);
    assert!(guarded(|| active.finish()).unwrap().can_retire());
    f.retire();
    assert_eq!(UNBOXED.with(Cell::get), unboxed + 1);
    assert_eq!(&*f.log.borrow(), &["payload", "observer", "custody"]);
}

#[test]
fn prepared_unactivated_observer_node_retires_without_native_or_retention_callback() {
    let f = Fixture::new(1);
    let ready = f.prepared();
    let unboxed = UNBOXED.with(Cell::get);
    drop(ready);
    f.retire();
    assert_eq!(UNBOXED.with(Cell::get), unboxed + 1);
    assert_eq!(f.calls.get(), 0);
    assert_eq!(f.observations.get(), 0);
    assert_eq!(f.retire_calls.get(), 0);
    assert_eq!(&*f.log.borrow(), &["custody"]);
}

#[test]
fn fixed_observer_refusals_return_once_and_never_release_stale_terminal_status() {
    for (mode, expected) in [
        (2, ScopedSubmissionProgress::Busy),
        (3, ScopedSubmissionProgress::NeedsFundedProgress),
        (4, ScopedSubmissionProgress::Unobservable),
    ] {
        let f = Fixture::new(mode);
        let active = f.prepared().activate(f.payload(), f.observer());
        let result = guarded(|| active.wait()).unwrap();
        assert_eq!(result.outcome, expected);
        assert!(
            result.status.settled,
            "native lifetime status is transported separately"
        );
        assert!(!result.can_retire());
        assert!(
            !f.settled.get(),
            "retention cannot publish stale completion"
        );
        assert_eq!(f.calls.get(), 1, "fixed refusal must not spin");
        assert_eq!(f.observations.get(), 1);
        let result = guarded(|| active.finish()).unwrap();
        assert_eq!(result.outcome, expected);
        assert!(
            f.log.borrow().is_empty(),
            "uncertain owner must enter quarantine"
        );
        f.mode.set(1);
        f.retire();
        assert_eq!(&*f.log.borrow(), &["payload", "observer", "custody"]);
    }
}

#[test]
fn native_observer_error_returns_actual_cause_and_retains_original_payload_until_later_proof() {
    let f = Fixture::new(5);
    let active = f.prepared().activate(f.payload(), f.observer());
    let unboxed = UNBOXED.with(Cell::get);
    let FinishRetainingError::Native(error) = guarded(|| active.finish()).unwrap_err() else { panic!("native cause expected"); };
    assert!(Rc::ptr_eq(&error.0, &f.cause));
    // The typed finish returns the actual cause. Its consuming Drop then runs
    // the existing guarded retirement attempt, whose error maps to a
    // conservative unsettled notification rather than successful observation.
    assert_eq!(f.calls.get(), 2);
    assert_eq!(f.observations.get(), 1);
    assert!(!f.settled.get());
    assert_eq!(f.retire_calls.get(), 0);
    assert_eq!(UNBOXED.with(Cell::get), unboxed);
    assert!(f.log.borrow().is_empty());
    f.mode.set(1);
    f.retire();
    assert_eq!(UNBOXED.with(Cell::get), unboxed + 1);
    assert!(Rc::ptr_eq(&error.0, &f.cause));
    assert_eq!(&*f.log.borrow(), &["payload", "observer", "custody"]);
    drop(error);
}

#[test]
fn observed_pending_waits_for_terminal_before_unboxing_and_releasing_custody() {
    let f = Fixture::new(6);
    let active = f.prepared().activate(f.payload(), f.observer());
    let unboxed = UNBOXED.with(Cell::get);
    let result = guarded(|| active.finish()).unwrap();
    assert!(result.can_retire());
    assert_eq!(f.calls.get(), 2);
    assert_eq!(f.observations.get(), 2);
    f.retire();
    assert_eq!(UNBOXED.with(Cell::get), unboxed + 1);
    assert_eq!(&*f.log.borrow(), &["payload", "observer", "custody"]);
}

#[test]
fn observer_acquisition_unwind_keeps_unstarted_node_custody_until_guarded_retirement() {
    let f = Fixture::new(1);
    let ready = f.prepared();
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _ = ready.try_activate::<()>(f.payload(), || panic!("acquisition fixture"));
    }));
    assert!(result.is_err());
    assert_eq!(f.calls.get(), 0);
    assert_eq!(f.observations.get(), 0);
    assert_eq!(&*f.log.borrow(), &["payload"]);
    f.retire();
    assert_eq!(&*f.log.borrow(), &["payload", "custody"]);
}

struct UnlockedPayload(Payload);
impl Retention for UnlockedPayload {
    fn observe(&self, status: Status) {
        self.0.observe(status);
    }
}
impl Drop for UnlockedPayload {
    fn drop(&mut self) {
        assert!(
            safemlx::can_reclaim_submission_resources(),
            "payload destructor must run outside native runtime guard"
        );
    }
}

#[test]
fn finish_retaining_keeps_same_node_through_unlocked_payload_drop_and_final_native_drain() {
    let f = Fixture::new(1);
    let ready =
        PreparedObservedRecovery::<UnlockedPayload, Custody, Fake>::new(Custody(f.log.clone()));
    let identity = ready.allocation_identity();
    let active = ready.activate(UnlockedPayload(f.payload()), f.observer());
    assert_eq!(active.allocation_identity(), identity);
    let unboxed = UNBOXED.with(Cell::get);
    let mut completed = guarded(|| active.finish_retaining()).unwrap();
    assert_eq!(UNBOXED.with(Cell::get), unboxed);
    assert_eq!(
        completed.cleanup.as_ref().unwrap().allocation_identity(),
        identity
    );
    assert!(f.log.borrow().is_empty());
    assert_eq!(completed.retention().0.token, 19);
    completed.retention_mut().0.token = 31;
    assert_eq!(completed.retention().0.token, 31);
    assert!(safemlx::can_reclaim_submission_resources());
    f.retirement.set(4);
    match completed.release() {
        Ok(observed) => assert!(observed.can_retire()),
        Err(error) => {
            // Other tests may own the runtime; payload is already outside it,
            // and the same empty node remains responsible for the final pass.
            assert!(error.pending.is_none());
            assert!(matches!(
                error.cause,
                FinishRetainingError::Observation(Observation {
                    outcome: ScopedSubmissionProgress::Busy,
                    ..
                })
            ));
        }
    }
    f.retire();
    assert_eq!(UNBOXED.with(Cell::get), unboxed + 1);
    assert_eq!(
        &*f.log.borrow(),
        &["payload", "retire", "observer", "custody"]
    );
}

#[test]
fn finish_retaining_never_transfers_payload_on_fixed_refusal_or_actual_native_error() {
    for mode in [2, 3, 4, 5] {
        let f = Fixture::new(mode);
        let active = f.prepared().activate(f.payload(), f.observer());
        let result = guarded(|| active.finish_retaining());
        match result {
            Err(FinishRetainingError::Native(cause)) if mode == 5 => {
                assert!(Rc::ptr_eq(&cause.0, &f.cause))
            }
            Err(FinishRetainingError::Observation(observed)) if mode != 5 => {
                assert!(!observed.can_retire())
            }
            _ => panic!("refusal cannot transfer payload"),
        }
        assert!(f.log.borrow().is_empty());
        f.mode.set(1);
        f.retire();
        assert_eq!(&*f.log.borrow(), &["payload", "observer", "custody"]);
    }
}

struct FailurePayload(Payload);
impl Retention for FailurePayload {
    fn observe(&self, status: Status) {
        assert!(status.failed && status.settled && !status.blocked);
        self.0.observations.set(self.0.observations.get() + 1);
    }
}

#[test]
fn terminal_native_failure_can_retire_but_cannot_transfer_success_payload() {
    let f = Fixture::new(7);
    let ready =
        PreparedObservedRecovery::<FailurePayload, Custody, Fake>::new(Custody(f.log.clone()));
    let active = ready.activate(FailurePayload(f.payload()), f.observer());
    let error = guarded(|| active.finish_retaining()).unwrap_err();
    let FinishRetainingError::Observation(observed) = error else {
        panic!("native failed status expected")
    };
    assert!(observed.status.failed && observed.can_retire());
    f.retire();
    assert!(
        f.retire_calls.get() >= 1,
        "failed but settled lifetime still needs exact native retirement"
    );
    assert_eq!(&*f.log.borrow(), &["payload", "observer", "custody"]);
}

#[test]
fn terminal_retirement_busy_and_actual_error_keep_node_until_successful_retry() {
    for retirement in [1, 2] {
        let f = Fixture::new(1);
        f.retirement.set(retirement);
        let ready = f.prepared();
        let unboxed = UNBOXED.with(Cell::get);
        let result = guarded(|| ready.activate(f.payload(), f.observer()).finish());
        match result {
            Ok(observed) if retirement == 1 => {
                assert_eq!(observed.outcome, ScopedSubmissionProgress::Busy);
                assert!(observed.status.settled && !observed.can_retire());
            }
            Err(FinishRetainingError::Native(cause)) if retirement == 2 => assert!(Rc::ptr_eq(&cause.0, &f.cause)),
            _ => panic!("actual retirement refusal expected"),
        }
        assert!(f.log.borrow().is_empty());
        assert_eq!(UNBOXED.with(Cell::get), unboxed);
        f.retirement.set(4);
        f.retire();
        assert_eq!(UNBOXED.with(Cell::get), unboxed + 1);
        assert_eq!(
            &*f.log.borrow(),
            &["retire", "payload", "retire", "observer", "custody"]
        );
    }
}

#[test]
fn completed_release_refuses_outer_guard_then_retains_same_empty_node_on_final_drain_failure() {
    for retirement in [1, 2] {
        let f = Fixture::new(1);
        let ready =
            PreparedObservedRecovery::<UnlockedPayload, Custody, Fake>::new(Custody(f.log.clone()));
        let identity = ready.allocation_identity();
        let unboxed = UNBOXED.with(Cell::get);
        let completed = guarded(|| {
            ready
                .activate(UnlockedPayload(f.payload()), f.observer())
                .finish_retaining()
        })
        .unwrap();
        let error = guarded(|| completed.release()).unwrap_err();
        let completed = error
            .pending
            .expect("outer guard returns untouched complete owner");
        assert_eq!(
            completed.cleanup.as_ref().unwrap().allocation_identity(),
            identity
        );
        assert_eq!(completed.retention().0.token, 19);
        assert!(f.log.borrow().is_empty());
        f.retirement.set(retirement);
        let error = completed.release().unwrap_err();
        assert!(
            error.pending.is_none(),
            "payload was destroyed; same empty node is quarantined"
        );
        match error.cause {
            FinishRetainingError::Observation(observed) => {
                assert_eq!(observed.outcome, ScopedSubmissionProgress::Busy)
            }
            FinishRetainingError::Native(cause) if retirement == 2 => {
                assert!(Rc::ptr_eq(&cause.0, &f.cause))
            }
            _ => panic!("actual final drain refusal expected"),
        }
        assert_eq!(&*f.log.borrow(), &["payload"]);
        assert_eq!(UNBOXED.with(Cell::get), unboxed);
        f.retirement.set(4);
        f.retire();
        assert_eq!(UNBOXED.with(Cell::get), unboxed + 1);
        assert_eq!(
            &*f.log.borrow(),
            &["payload", "retire", "observer", "custody"]
        );
    }
}

#[test]
fn terminal_retirement_callback_panic_fences_retry_and_keeps_node_custody() {
    let f = Fixture::new(1);
    f.retirement.set(3);
    let ready = f.prepared();
    let unboxed = UNBOXED.with(Cell::get);
    assert!(catch_unwind(AssertUnwindSafe(|| {
        guarded(|| ready.activate(f.payload(), f.observer()).finish())
    }))
    .is_err());
    assert_eq!(f.retire_calls.get(), 1);
    assert!(f.log.borrow().is_empty());
    f.retirement.set(0);
    guarded(reap);
    assert_eq!(
        f.retire_calls.get(),
        1,
        "failed callback must never run again"
    );
    assert_eq!(UNBOXED.with(Cell::get), unboxed);
    assert!(
        f.log.borrow().is_empty(),
        "unobservable callback retains payload and custody"
    );
}

#[test]
fn release_with_keeps_uncalled_owned_callback_on_boundary_refusal_then_runs_before_final_drain() {
    let f = Fixture::new(1);
    let called = Rc::new(Cell::new(false));
    let callback_owner = Rc::new(());
    let callback = {
        let called = called.clone();
        let owner = callback_owner.clone();
        move |payload: &mut Payload| {
            assert!(safemlx::can_reclaim_submission_resources());
            assert_eq!(Rc::strong_count(&owner), 2);
            assert_eq!(payload.token, 19);
            payload.token = 41;
            called.set(true);
        }
    };
    let completed = guarded(|| {
        f.prepared()
            .activate(f.payload(), f.observer())
            .finish_retaining()
    })
    .unwrap();
    let error = guarded(|| completed.release_with(callback)).unwrap_err();
    assert!(!called.get());
    assert_eq!(Rc::strong_count(&callback_owner), 2);
    assert!(f.log.borrow().is_empty());
    let result = error.pending.unwrap().release_with(error.callback.unwrap());
    assert!(called.get());
    assert_eq!(Rc::strong_count(&callback_owner), 1);
    if let Err(error) = result {
        assert!(error.callback.is_none() && error.pending.is_none());
        assert!(matches!(
            error.cause,
            FinishRetainingError::Observation(Observation {
                outcome: ScopedSubmissionProgress::Busy,
                ..
            })
        ));
    }
    f.retire();
    assert_eq!(&*f.log.borrow(), &["payload", "observer", "custody"]);
}

#[test]
fn release_callback_unwind_preserves_same_cleanup_node_until_later_guarded_retirement() {
    let f = Fixture::new(1);
    let completed = guarded(|| {
        f.prepared()
            .activate(f.payload(), f.observer())
            .finish_retaining()
    })
    .unwrap();
    let unboxed = UNBOXED.with(Cell::get);
    let retire_calls = f.retire_calls.get();
    assert!(catch_unwind(AssertUnwindSafe(|| {
        let _ = completed.release_with(|_: &mut Payload| panic!("targeted release fixture"));
    }))
    .is_err());
    assert_eq!(UNBOXED.with(Cell::get), unboxed);
    assert_eq!(
        f.retire_calls.get(),
        retire_calls,
        "unwinding cannot run native retirement callback"
    );
    assert!(
        f.log.borrow().is_empty(),
        "abandoned T returns to same cleanup node"
    );
    f.retire();
    assert_eq!(UNBOXED.with(Cell::get), unboxed + 1);
    assert_eq!(&*f.log.borrow(), &["payload", "observer", "custody"]);
}

#[test]
fn consuming_refusal_maps_actual_cause_before_observer_can_disappear() {
    for mode in [2, 3, 4] {
        let f = Fixture::new(mode);
        f.map_error.set(true);
        let error = guarded(|| {
            f.prepared()
                .activate(f.payload(), f.observer())
                .finish_retaining()
        })
        .unwrap_err();
        let FinishRetainingError::Native(cause) = error else {
            panic!("fixed native cause expected")
        };
        assert!(Rc::ptr_eq(&cause.0, &f.cause));
        assert!(f.log.borrow().is_empty());
        f.mode.set(1);
        f.retire();
        assert!(Rc::ptr_eq(&cause.0, &f.cause));
    }
    let f = Fixture::new(7);
    f.map_error.set(true);
    let ready =
        PreparedObservedRecovery::<FailurePayload, Custody, Fake>::new(Custody(f.log.clone()));
    let FinishRetainingError::Native(cause) = guarded(|| {
        ready
            .activate(FailurePayload(f.payload()), f.observer())
            .finish()
    })
    .unwrap_err() else { panic!("native cause expected"); };
    assert!(Rc::ptr_eq(&cause.0, &f.cause));
    f.retire();
    assert!(Rc::ptr_eq(&cause.0, &f.cause));
}

struct NestedResource {
    log: Log,
    retirement: Rc<Cell<u8>>,
}
impl Drop for NestedResource {
    fn drop(&mut self) {
        assert!(safemlx::can_reclaim_submission_resources());
        self.log.borrow_mut().push("nested");
        // Model a wrapper which now needs a later exact-owner native drain.
        self.retirement.set(1);
    }
}
struct NestedSlot {
    _value: NestedResource,
    cleanup: Option<OriginalRetirementCleanup>,
    _custody: NestedCustody,
}
struct NestedCustody(Log);
impl Drop for NestedCustody {
    fn drop(&mut self) {
        self.0.borrow_mut().push("nested-custody");
    }
}
struct NestedPayload {
    resources: Option<crate::backend::ordinary_retirement::OrdinaryRetirement<NestedSlot>>,
}
impl Retention for NestedPayload {
    fn observe(&self, _: Status) {}
    fn retire_original(mut self, cleanup: OriginalRetirementCleanup) {
        let mut resources = self.resources.take().expect("prepared nested payload");
        resources.cleanup = Some(cleanup);
        drop(resources); // queue the exact existing payload Box, no callback.
    }
}

#[test]
fn refused_completed_drop_preserves_cleanup_after_actual_nested_payload_destruction() {
    let f = Fixture::new(1);
    let payload = NestedPayload {
        resources: Some(
            crate::backend::ordinary_retirement::OrdinaryRetirement::new(NestedSlot {
                _value: NestedResource {
                    log: f.log.clone(),
                    retirement: f.retirement.clone(),
                },
                cleanup: None,
                _custody: NestedCustody(f.log.clone()),
            }),
        ),
    };
    let ready =
        PreparedObservedRecovery::<NestedPayload, Custody, Fake>::new(Custody(f.log.clone()));
    let unboxed = UNBOXED.with(Cell::get);
    let completed = guarded(|| ready.activate(payload, f.observer()).finish_retaining()).unwrap();
    let calls = f.calls.get();
    let retired = f.retire_calls.get();
    guarded(|| {
        let refused = completed.release().unwrap_err();
        assert!(refused.pending.is_some());
        drop(refused); // must restore T and queue same node without callbacks.
    });
    assert_eq!((f.calls.get(), f.retire_calls.get()), (calls, retired));
    assert!(f.log.borrow().is_empty());
    guarded(reap); // terminal handoff moves cleanup into the nested payload Box.
    assert_eq!(UNBOXED.with(Cell::get), unboxed);
    assert!(f.log.borrow().is_empty());
    crate::backend::ordinary_retirement::reclaim();
    assert_eq!(&*f.log.borrow(), &["nested", "nested-custody"]);
    guarded(reap); // actual resource is gone, but native drain still refuses.
    assert_eq!(UNBOXED.with(Cell::get), unboxed);
    assert_eq!(&*f.log.borrow(), &["nested", "nested-custody"]);
    f.retirement.set(4);
    f.retire();
    assert_eq!(UNBOXED.with(Cell::get), unboxed + 1);
    assert_eq!(
        &*f.log.borrow(),
        &["nested", "nested-custody", "retire", "observer", "custody"]
    );
}

#[test]
fn retained_child_finishes_once_after_pending_busy_and_retirement_error() {
    let fixture = Fixture::new(0);
    fixture.map_error.set(true);
    let mut child = fixture
        .prepared()
        .activate(fixture.payload(), fixture.observer());
    let identity = child.allocation_identity();
    assert!(matches!(child.try_finish_successfully().unwrap(), RetirementAttempt::Pending));
    assert_eq!(child.allocation_identity(), identity);
    assert_eq!(fixture.retire_calls.get(), 0);

    fixture.mode.set(2); // Actual observer contention keeps the same child.
    assert!(matches!(child.try_finish_successfully().unwrap(), RetirementAttempt::Pending));
    fixture.mode.set(1);
    fixture.retirement.set(1); // Native registry contention is independent.
    assert!(matches!(child.try_finish_successfully().unwrap(), RetirementAttempt::Pending));
    assert_eq!(child.allocation_identity(), identity);
    assert!(fixture.log.borrow().is_empty());

    fixture.retirement.set(2);
    let error = child.try_finish_successfully().unwrap_err();
    assert!(Rc::ptr_eq(&error.0, &fixture.cause));
    drop(error);
    assert_eq!(child.allocation_identity(), identity);
    assert!(fixture.log.borrow().is_empty());

    fixture.retirement.set(4);
    assert!(matches!(child.try_finish_successfully().unwrap(), RetirementAttempt::Retired));
    assert_eq!(
        fixture
            .log
            .borrow()
            .iter()
            .filter(|entry| **entry == "payload")
            .count(),
        1
    );
    drop(child);
    fixture.retire();
    let log = fixture.log.borrow();
    assert_eq!(log.iter().filter(|entry| **entry == "payload").count(), 1);
    assert_eq!(log.iter().filter(|entry| **entry == "custody").count(), 1);
    assert_eq!(log.iter().filter(|entry| **entry == "observer").count(), 1);
    assert!(
        log.iter().position(|entry| *entry == "payload").unwrap()
            < log.iter().position(|entry| *entry == "custody").unwrap()
    );
}

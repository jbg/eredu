//! The same real core step driver for ordinary, controlled and freshly resumed runs.
use super::*;

enum Run<'a> {
    Ordinary(TextGeneration<'a, Backend>),
    Controlled(ControlledTextGeneration<'a, Backend, Controller>),
}
impl Run<'_> {
    fn advance(
        &mut self,
        cancel: &crate::GenerationCancellationToken,
    ) -> Option<Result<(), String>> {
        match self {
            Self::Ordinary(run) => run
                .next_cancellable(cancel)
                .map(|value| value.map(|_| ()).map_err(|e| e.to_string())),
            Self::Controlled(run) => run
                .next_cancellable(cancel)
                .map(|value| value.map(|_| ()).map_err(|e| e.to_string())),
        }
    }
    fn context(&self) -> &TextStepContext {
        match self {
            Self::Ordinary(run) => &run.inner.step_context,
            Self::Controlled(run) => &run.inner.step_context,
        }
    }
    fn remaining(&self) -> Option<usize> {
        match self {
            Self::Ordinary(run) => run.inner.remaining_tokens,
            Self::Controlled(run) => run.inner.remaining_tokens,
        }
    }
    fn changes(&self) -> usize {
        match self {
            Self::Ordinary(run) => run.inner.backend_state.changes,
            Self::Controlled(run) => run.inner.backend_state.changes,
        }
    }
    fn change_session_admission(&mut self) {
        let runtime = match self {
            Self::Ordinary(run) => &mut *run.runtime,
            Self::Controlled(run) => &mut *run.runtime,
        };
        runtime.session_mut().capabilities =
            SessionCapabilities::default().with_activation_inspection(true);
    }
}
fn start<'a>(
    runtime: &'a mut ModelRuntime<Backend>,
    facts: &Rc<RefCell<Facts>>,
    saved: &Saved,
    cancel: &crate::GenerationCancellationToken,
    resumed: bool,
    controlled: bool,
) -> Run<'a> {
    if resumed {
        let mut f = facts.borrow_mut();
        f.resume.enabled = true;
        f.resume.cancellation = Some(cancel.clone());
    }
    match (resumed, controlled) {
        (false, false) => {
            Run::Ordinary(TextGeneration::new(runtime, vec![1, 2], limited(4)).unwrap())
        }
        (false, true) => Run::Controlled(
            ControlledTextGeneration::new(
                runtime,
                vec![1, 2],
                limited(4),
                controller(facts, ControllerFailure::None),
            )
            .unwrap(),
        ),
        (true, false) => Run::Ordinary(
            TextGeneration::resume_saved(runtime, saved, limited(4), cancel)
                .unwrap()
                .unwrap(),
        ),
        (true, true) => Run::Controlled(
            ControlledTextGeneration::resume_saved(
                runtime,
                saved,
                limited(4),
                controller(facts, ControllerFailure::None),
                cancel,
            )
            .unwrap()
            .unwrap(),
        ),
    }
}
fn clear_step_events(facts: &Rc<RefCell<Facts>>) {
    let mut f = facts.borrow_mut();
    f.events.clear();
    f.votes.clear();
}
fn assert_terminal(run: &mut Run<'_>, facts: &Rc<RefCell<Facts>>) {
    let before = (
        facts.borrow().events.clone(),
        facts.borrow().votes.clone(),
        facts.borrow().contexts.len(),
        run.context().clone(),
    );
    for _ in 0..2 {
        assert!(run
            .advance(&crate::GenerationCancellationToken::new())
            .is_none());
    }
    assert_eq!(
        (
            facts.borrow().events.clone(),
            facts.borrow().votes.clone(),
            facts.borrow().contexts.len(),
            run.context().clone()
        ),
        before
    );
    assert!(!facts.borrow().active);
}

#[test]
fn local_cancellation_precedes_permit_controller_and_submission_for_all_routes() {
    for resumed in [false, true] {
        for controlled in [false, true] {
            for decode in [false, true] {
                let (mut runtime, facts) = fixture();
                let saved = source();
                let cancel = crate::GenerationCancellationToken::new();
                let old_context = saved.old_context.clone();
                let words = saved.words.clone();
                let old_owners = Rc::strong_count(&saved.old_preparation.0);
                let mut run = start(&mut runtime, &facts, &saved, &cancel, resumed, controlled);
                if decode {
                    run.advance(&cancel).unwrap().unwrap();
                }
                let before = run.context().clone();
                let remaining = run.remaining();
                let changes = run.changes();
                let issued = facts.borrow().contexts.len();
                clear_step_events(&facts);
                cancel.cancel();
                assert!(run.advance(&cancel).is_none());
                assert_eq!(run.context(), &before);
                assert_eq!(run.remaining(), remaining);
                assert_eq!(run.changes(), changes);
                assert_eq!(facts.borrow().contexts.len(), issued);
                assert_eq!(
                    facts.borrow().votes,
                    [(Stage::Prediction, Status::Cancelled)]
                );
                assert_eq!(
                    facts.borrow().events,
                    if decode && !controlled {
                        vec!["wait"]
                    } else {
                        vec![]
                    }
                );
                assert_eq!(facts.borrow().outstanding_completions, 0);
                assert_eq!(saved.old_context, old_context);
                assert!(Rc::ptr_eq(&saved.words, &words));
                assert_eq!(Rc::strong_count(&saved.old_preparation.0), old_owners);
                assert_terminal(&mut run, &facts);
            }
        }
    }
}

#[test]
fn cancelled_lane_and_ready_peer_both_stop_but_issued_peer_attempt_is_not_refunded() {
    for resumed in [false, true] {
        for controlled in [false, true] {
            for decode in [false, true] {
                for local_cancel in [true, false] {
                    let (mut runtime, facts) = fixture();
                    let saved = source();
                    let cancel = crate::GenerationCancellationToken::new();
                    let mut run = start(&mut runtime, &facts, &saved, &cancel, resumed, controlled);
                    if decode {
                        run.advance(&cancel).unwrap().unwrap();
                    }
                    let before = run.context().clone();
                    let remaining = run.remaining();
                    let issued = facts.borrow().contexts.len();
                    clear_step_events(&facts);
                    facts.borrow_mut().peer_cancel = Some(Stage::Prediction);
                    if local_cancel {
                        cancel.cancel();
                    }
                    assert!(run.advance(&cancel).is_none());
                    assert!(cancel.is_cancelled());
                    assert_eq!(run.context().run_identity(), before.run_identity());
                    assert_eq!(run.context().policy_identity(), before.policy_identity());
                    assert_eq!(
                        run.context().attempt(),
                        before.attempt() + u64::from(!local_cancel)
                    );
                    assert_eq!(run.remaining(), remaining);
                    assert_eq!(
                        facts.borrow().contexts.len(),
                        issued + usize::from(!local_cancel)
                    );
                    assert_eq!(
                        facts.borrow().votes,
                        [(
                            Stage::Prediction,
                            if local_cancel {
                                Status::Cancelled
                            } else {
                                Status::Ready
                            }
                        )]
                    );
                    let mut events = if decode && !controlled {
                        vec!["wait"]
                    } else {
                        vec![]
                    };
                    if !local_cancel {
                        events.extend(["begin", "abort"]);
                    }
                    assert_eq!(facts.borrow().events, events);
                    assert_eq!(facts.borrow().outstanding_completions, 0);
                    assert_terminal(&mut run, &facts);
                }
            }
        }
    }
}

#[test]
fn outstanding_completion_failure_takes_precedence_over_local_or_peer_cancellation() {
    for resumed in [false, true] {
        for local_cancel in [true, false] {
            let (mut runtime, facts) = fixture();
            let saved = source();
            let cancel = crate::GenerationCancellationToken::new();
            let mut run = start(&mut runtime, &facts, &saved, &cancel, resumed, false);
            run.advance(&cancel).unwrap().unwrap();
            assert_eq!(facts.borrow().outstanding_completions, 1);
            let before = run.context().clone();
            clear_step_events(&facts);
            {
                let mut f = facts.borrow_mut();
                f.fail_completion = true;
                f.peer_cancel = Some(Stage::Prediction);
            }
            if local_cancel {
                cancel.cancel();
            }
            let error = run.advance(&cancel).unwrap().unwrap_err();
            assert!(error.contains("completion settled with failure"));
            assert_eq!(run.context(), &before);
            assert_eq!(facts.borrow().votes, [(Stage::Prediction, Status::Failed)]);
            assert_eq!(facts.borrow().events, ["wait"]);
            assert_eq!(facts.borrow().outstanding_completions, 0);
            assert_terminal(&mut run, &facts);
        }
    }
}

#[test]
fn changed_session_admission_is_a_failure_even_when_locally_cancelled() {
    for resumed in [false, true] {
        for controlled in [false, true] {
            let (mut runtime, facts) = fixture();
            let saved = source();
            let cancel = crate::GenerationCancellationToken::new();
            let mut run = start(&mut runtime, &facts, &saved, &cancel, resumed, controlled);
            let before = run.context().clone();
            clear_step_events(&facts);
            run.change_session_admission();
            cancel.cancel();
            assert!(run
                .advance(&cancel)
                .unwrap()
                .unwrap_err()
                .contains("changed session admission"));
            assert_eq!(run.context(), &before);
            assert_eq!(facts.borrow().votes, [(Stage::Prediction, Status::Failed)]);
            assert!(facts.borrow().events.is_empty());
            assert_terminal(&mut run, &facts);
        }
    }
}

#[test]
fn cancellation_never_masks_prediction_agreement_rejection_or_transport_failure() {
    for transport in [false, true] {
        for controlled in [false, true] {
            let (mut runtime, facts) = fixture();
            let saved = source();
            let cancel = crate::GenerationCancellationToken::new();
            let mut run = start(&mut runtime, &facts, &saved, &cancel, false, controlled);
            let before = run.context().clone();
            clear_step_events(&facts);
            cancel.cancel();
            if transport {
                facts.borrow_mut().fail_agreement = Some(Stage::Prediction);
            } else {
                facts.borrow_mut().peer_reject = Some(Stage::Prediction);
            }
            let error = run.advance(&cancel).unwrap().unwrap_err();
            assert!(error.contains(if transport {
                "preparation transport failed"
            } else {
                "rejected by rank 1"
            }));
            assert_eq!(run.context(), &before);
            assert!(facts.borrow().events.is_empty());
            assert_eq!(
                facts.borrow().votes,
                [(Stage::Prediction, Status::Cancelled)]
            );
            assert_terminal(&mut run, &facts);
        }
    }
}
